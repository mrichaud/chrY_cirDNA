import re
import pandas as pd
import numpy as np
import sys
import re
import os
from Bio.Seq import Seq
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib import colormaps
from scipy import stats

#-------------
# 1. Charger les rÃ©gions centromÃ©riques
#-------------
def load_centromere_regions(centromere_file):
    centromeres = {}

    with open(centromere_file) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue

            parts = line.split()
            chrom = parts[0]
            # VÃ©rifier que toutes les coordonnÃ©es sont numÃ©riques
            if not all(p.isdigit() for p in parts[1:]): 
                print(f"âš ï¸ Ligne ignorÃ©e (non numÃ©rique) : {line}")
                continue
            
            coords = list(map(int, parts[1:]))

            # Si nombre impair â†’ on ignore la derniÃ¨re coordonnÃ©e
            if len(coords) % 2 != 0:
                coords = coords[:-1]

            # Regrouper par paires (start, end)
            intervals = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

            centromeres.setdefault(chrom, []).extend(intervals)

    return centromeres


#-------------
# 2. Regarder si bin est dans une rÃ©gion centromÃ©rique
#-------------
def is_in_centromere(chrom, start, end, centromeres):
    """
    VÃ©rifie si une rÃ©gion chevauche une rÃ©gion centromÃ©rique.
    """
    if str(chrom) not in centromeres:
        return False
    
    for centro_start, centro_end in centromeres[str(chrom)]:
        # VÃ©rifie s'il y a un chevauchement
        if not (end <= centro_start or start >= centro_end):
            return True
    return False


#-------------
# 1. Nucleosome count
#-------------

def load_nucleosome_count_table(tsv_dir, centromere_file=None):
    """
    Charge tous les TSV gÃ©nÃ©rÃ©s (comptage) et les concatÃ¨ne.
    """
    tsv_files = list(Path(tsv_dir).glob("*nucleosome_counts_*global*.tsv"))

    if not tsv_files:
        raise FileNotFoundError(f"Aucun fichier TSV trouvÃ© dans {tsv_dir}")


    df = pd.concat((pd.read_csv(f, sep="\t", dtype={"chromosome": str}).assign(source_file=f.name)
         for f in tsv_files),ignore_index=True)

    if centromere_file:
        centromeres = load_centromere_regions(centromere_file)
        print("Avant filtrage :", len(df))

        # Masque vectorisÃ©
        df["in_centromere"] = df.apply(lambda r: is_in_centromere(r["chromosome"], r["start"], r["end"], centromeres), axis=1)
        df_long = df[~df["in_centromere"]].drop(columns="in_centromere").reset_index(drop=True)

        print("AprÃ¨s filtrage :", len(df_long))

    return df_long


# crÃ©er un barplot du nombre de nuclÃ©osome par chromosome en utilisant le DataFrame long crÃ©Ã© prÃ©cÃ©dement
def plot_barplot_nucleosome_count(df_long, suffix):
    """Barplot du nombre de nuclÃ©osomes par chromosome"""
    
    sns.set_theme(style="whitegrid")
    
    df_long["chromosome"] = df_long["chromosome"].astype(str)

    # ORDRE : 1-22, puis X, Y
    chrom_order = sorted([c for c in df_long["chromosome"].unique() if c not in ["X", "Y"]], key=lambda x: int(x) if x.isdigit() else 99) + ["X", "Y"]
    
    df_long["chromosome"] = pd.Categorical(df_long["chromosome"], categories=chrom_order, ordered=True)
    
    # Palette : bleu pour autosomes (1-22), vert pour X et Y
    colors = ["#5975A4" if c not in ["X", "Y"] else "#5F9E6E" 
              for c in chrom_order]
    
     #calcul de l'IC95% pour chaque chromosome
    stats_by_chrom = []
    for chrom in chrom_order:
        chrom_data = df_long[df_long["chromosome"] == chrom]["nucleosome_count"]
        
        mean = chrom_data.mean()
        std = chrom_data.std(ddof=1)  # ddof=1 pour Ã©chantillon
        n = len(chrom_data)
        ic95 = 1.96 * std / np.sqrt(n)  # IC95%
        
        stats_by_chrom.append({
            "chromosome": chrom,
            "mean": mean,
            "ic95": ic95,
            "std": std,
            "n": n})
    
    df_stats = pd.DataFrame(stats_by_chrom)
    
    plt.figure(figsize=(14, 8))

    # Barplot avec CI Ã  95%
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Barplot seaborn avec estimator=mean (sans barres d'erreur automatiques)
    sns.barplot(
        data=df_long,
        x="chromosome",
        y="nucleosome_count",
        palette=colors,
        hue="chromosome",
        dodge=False,
        errorbar=None,  # DÃ©sactiver les barres d'erreur auto
        ax=ax,
        edgecolor="black",
        linewidth=0.8)
    
    # Ajout des barres d'erreur IC95%
    x_pos = range(len(df_stats))
    ax.errorbar(
        x=x_pos,
        y=df_stats["mean"],
        yerr=df_stats["ic95"],
        fmt='none',
        ecolor='black',
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        zorder=10)
    
    ax.set_xlabel("Chromosomes", fontsize=12, fontweight='bold')
    ax.set_ylabel("Nombre de nuclÃ©osomes", fontsize=12, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title("Nombre de nuclÃ©osomes par chromosome (bins de 10kb)\n"
                 "Barres d'erreur : IC95% (Â±1.96 Ã— SE)", 
                 fontsize=14,
                 fontweight='bold',
                 pad=15)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    legend_elements = [
        Patch(facecolor='#5975A4', edgecolor='black', label='Autosome'),
        Patch(facecolor='#5F9E6E', edgecolor='black', label='Sex'),
        Patch(facecolor='white', edgecolor='white', label='*calculÃ© en enlevant \n les rÃ©gions centromÃ©riques')]
    
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(1.05, 1.1),
        fontsize=7,
        frameon=True,
        shadow=True,
        fancybox=True,
        framealpha=0.95,
        title="LÃ©gende",
        title_fontsize=8)
    
    plt.tight_layout()
  
    out = f"barplot_nucleosome_counts_IC95_{suffix}_sanslog_bins10kb.svg"
    plt.savefig(out, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


#-------------
# 2. Average distance between nucleosomes (log2)
#-------------

def load_average_distance_table(tsv_dir, centromere_file=None):
    """
    Charge tous les TSV gÃ©nÃ©rÃ©s (distance) et les concatÃ¨ne.
    """
    tsv_files = list(Path(tsv_dir).glob("*average_distance_*global*.tsv"))

    if not tsv_files:
        raise FileNotFoundError(f"Aucun fichier TSV trouvÃ© dans {tsv_dir}")

    df = pd.concat((pd.read_csv(f, sep="\t", dtype={"chromosome": str}).assign(source_file=f.name)
         for f in tsv_files),ignore_index=True)

    if centromere_file:
        centromeres = load_centromere_regions(centromere_file)
        print("Avant filtrage :", len(df))

        # Masque vectorisÃ©
        df["in_centromere"] = df.apply(lambda r: is_in_centromere(r["chromosome"], r["start"], r["end"], centromeres), axis=1)
        df_long_1 = df[~df["in_centromere"]].drop(columns="in_centromere").reset_index(drop=True)

        print("AprÃ¨s filtrage :", len(df_long_1))

    return df_long_1
 


# crÃ©er un barplot de la distance au nuclÃ©osome par chromosome en utilisant le DataFrame long crÃ©Ã© prÃ©cÃ©dement

def plot_barplot_average_distance(df_long, suffix):
    """Barplot du nombre de nuclÃ©osomes par chromosome"""
    
    sns.set_theme(style="whitegrid")
    
    df_long["chromosome"] = df_long["chromosome"].astype(str)

    # ORDRE : 1-22, puis X, Y
    chrom_order = sorted([c for c in df_long["chromosome"].unique() if c not in ["X", "Y"]], key=lambda x: int(x) if x.isdigit() else 99) + ["X", "Y"]
    
    df_long["chromosome"] = pd.Categorical(df_long["chromosome"], categories=chrom_order, ordered=True)
    
    # Palette : bleu pour autosomes (1-22), vert pour X et Y
    colors = ["#5975A4" if c not in ["X", "Y"] else "#5F9E6E" 
              for c in chrom_order]
    
    #calcul de l'IC95% pour chaque chromosome
    stats_by_chrom = []
    for chrom in chrom_order:
        chrom_data = df_long[df_long["chromosome"] == chrom]["average_distance"]
        
        mean = chrom_data.mean()
        std = chrom_data.std(ddof=1)  # ddof=1 pour Ã©chantillon
        n = len(chrom_data)
        ic95 = 1.96 * std / np.sqrt(n)  # IC95%
        
        stats_by_chrom.append({
            "chromosome": chrom,
            "mean": mean,
            "ic95": ic95,
            "std": std,
            "n": n})
    
    df_stats = pd.DataFrame(stats_by_chrom)
    
    plt.figure(figsize=(14, 8))

    # Barplot avec CI Ã  95%
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Barplot seaborn avec estimator=mean (sans barres d'erreur automatiques)
    sns.barplot(
        data=df_long,
        x="chromosome",
        y="average_distance",
        palette=colors,
        hue="chromosome",
        dodge=False,
        errorbar=None,  # DÃ©sactiver les barres d'erreur auto
        ax=ax,
        edgecolor="black",
        linewidth=0.8)
    
    # Ajout des barres d'erreur IC95%
    x_pos = range(len(df_stats))
    ax.errorbar(
        x=x_pos,
        y=df_stats["mean"],
        yerr=df_stats["ic95"],
        fmt='none',
        ecolor='black',
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        zorder=10)

    ax.set_xlabel("Chromosomes", fontsize=12, fontweight='bold')
    ax.set_ylabel("Distance moyenne entre nuclÃ©osomes", fontsize=12, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title("Distance moyenne entre nuclÃ©osomes par chromosome (bins de 10kb)\n"
                 "Barres d'erreur : IC95% (Â±1.96 Ã— SE)",
                 fontsize=14,
                 fontweight='bold',
                 pad=15)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    legend_elements = [
        Patch(facecolor='#5975A4', edgecolor='black', label='Autosome'),
        Patch(facecolor='#5F9E6E', edgecolor='black', label='Sex'),
        Patch(facecolor='white', edgecolor='white', label='*calculÃ© en enlevant \n les rÃ©gions centromÃ©riques')]
    
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(1.05, 1.1),
        fontsize=7,
        frameon=True,
        shadow=True,
        fancybox=True,
        framealpha=0.95,
        title="LÃ©gende",
        title_fontsize=8)

    
    plt.tight_layout()
  
    out = f"barplot_average_distance_IC95_{suffix}sanslog_bins10kb.svg"
    plt.savefig(out, dpi=300, facecolor='white', edgecolor='none',bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")
    
#-------------
# 3. Main program
#-------------


if __name__ == "__main__":

    txt_files = [Path("/data2/jcolinge/fragmentomics/atlas/cris-healthy-wps-peaks/compiled-selection-with-Y.txt")]
    centromere_file = "/data2/USERS/lruffinel/cfDNA/chromosomes_cover_uncover/centromere-region.txt"

    tsv_dir = Path("/data2/USERS/richaud/chrY/distance_nucleosome_sanslog/")
    tsv_dir.mkdir(exist_ok=True)

    # 1) Charger tous les TSV
    df_long = load_nucleosome_count_table(tsv_dir, centromere_file=centromere_file)
    df_long.to_csv("df_long_nucleosome_count_global_sans_centromere.csv", index=False)

    df_long_1 = load_average_distance_table(tsv_dir, centromere_file=centromere_file)
    df_long_1.to_csv("df_long_average_distance_global_sans_centromere.csv", index=False)

    # 2) Tracer les plots
    plot_barplot_nucleosome_count(df_long, "global")

    plot_barplot_average_distance(df_long_1, "global")
