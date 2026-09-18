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
import matplotlib.colors as mcolors
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib import colormaps
from scipy import stats



#-------------
# 1. Nucleosome count
#-------------

def load_nucleosome_count_table(tsv_dir):
    """
    Charge tous les TSV générés (comptage) et les concatène.
    """
    tsv_files = list(Path(tsv_dir).glob("*nucleosome_counts_prostate_sans_PAR.tsv"))

    if not tsv_files:
        raise FileNotFoundError(f"Aucun fichier TSV trouvé dans {tsv_dir}")

    dfs = []
    for f in tsv_files:
       df = pd.read_csv(f, sep="\t", dtype={"chromosome": str})
       df["source_file"] = f.name
       dfs.append(df)

    df_long = pd.concat(dfs, ignore_index=True)
    return df_long


def load_centromere_regions(centromere_file):
    """
    Charge le fichier centromere-region.txt.
    Retourne un dict : {chrom: [(start, end), ...]}
    """
    centro = {}
    with open(centromere_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] == "chr":
                continue
            chrom = parts[0]
            # Le Y a plusieurs intervalles sur une seule ligne
            intervals = []
            i = 1
            while i + 1 < len(parts):
                start = int(parts[i])
                end = int(parts[i + 1])
                if end > 0:  # ignorer les intervalles vides (start=0, end=0)
                    intervals.append((start, end))
                i += 2
            if intervals:
                centro[chrom] = intervals
    return centro


def plot_ideogram_nucleosome_count(df_long, suffix, centromere_file=None):

    # 1. PRÉPARATION DES DONNÉES
    df_long["chromosome"] = df_long["chromosome"].astype(str)
    print(df_long)
    df_long = df_long.sort_values(["chromosome", "start"])

    # 2. ORDRE DES CHROMOSOMES
    chroms = sorted([c for c in df_long["chromosome"].unique() if c.isdigit()], key=int)
    chroms += [c for c in ["X", "Y"] if c in df_long["chromosome"].unique()]

    n_chroms = len(chroms)

    # 3. COLORMAP ET NORMALISATION
    cmap = plt.cm.RdYlBu_r
    vmin = df_long["nucleosome_count"].quantile(0.05)
    vmax = df_long["nucleosome_count"].quantile(0.95)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # 4. CHARGEMENT DES CENTROMÈRES
    centro = {}
    if centromere_file is not None:
        centro = load_centromere_regions(centromere_file)

    # 5. FIGURE : un axe par chromosome
    fig, axes = plt.subplots(
        n_chroms, 1,
        figsize=(20, n_chroms * 0.55),
        gridspec_kw={"hspace": 0.25}
    )
    fig.subplots_adjust(left=0.08, top=0.97)

    if n_chroms == 1:
        axes = [axes]

    for ax, chrom in zip(axes, chroms):
        data = df_long[df_long["chromosome"] == chrom].sort_values("start")

        values = data["nucleosome_count"].values
        n_bins = len(values)
        img = values.reshape(1, -1)

        ax.imshow(
            img,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest"
        )

        # 6. SUPERPOSITION DES RÉGIONS CENTROMÉRIQUES EN GRIS
        if chrom in centro:
            chrom_start = data["start"].min()
            chrom_end = data["end"].max()
            chrom_len = chrom_end - chrom_start

            for (c_start, c_end) in centro[chrom]:
                # Convertir coordonnées génomiques → coordonnées pixel (bins)
                x0 = (c_start - chrom_start) / chrom_len * n_bins - 0.5
                x1 = (c_end   - chrom_start) / chrom_len * n_bins - 0.5
                ax.add_patch(plt.Rectangle(
                    (x0, -0.5),          # (x, y) coin bas-gauche
                    x1 - x0,            # largeur
                    1.0,                 # hauteur (toute la bande)
                    linewidth=0,
                    facecolor="grey",
                    alpha=0.85,
                    zorder=2             # par-dessus l'heatmap
                ))

        ax.set_yticks([0])
        ax.set_yticklabels([f"{chrom}"], fontsize=20, va="center")
        ax.set_xticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    # 7. LABEL Y
    fig.text(0.04, 0.5, "Chromosomes", fontsize=24, fontweight='bold',
             va='center', ha='center', rotation='vertical')

    # 8. COLORBAR HORIZONTALE EN HAUT
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.15, 1.00, 0.70, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("Nombre de nucléosomes", fontsize=22, fontweight='bold', labelpad=8)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")

    # 9. LÉGENDE CENTROMÈRE
    if centromere_file is not None:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='grey', alpha=0.85, label='Centromère')]
        fig.legend(
            handles=legend_elements,
            loc='lower right',
            bbox_to_anchor=(0.98, 0.01),
            fontsize=16,
            frameon=True
        )

    # 10. SAUVEGARDER
    plt.savefig(
        f"ideogram_nucleosome_count_{suffix}_prostate.svg",
        dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close()

    print(f"Saved ideogram_nucleosome_count_{suffix}_prostate.svg")


#-------------
# 2. Average distance between nucleosomes (log2)
#-------------

def load_average_distance_table(tsv_dir):
    """
    Charge tous les TSV générés (distance) et les concatène.
    """
    tsv_files = list(Path(tsv_dir).glob("*average_distance_prostate_sans_PAR.tsv"))

    if not tsv_files:
        raise FileNotFoundError(f"Aucun fichier TSV trouvé dans {tsv_dir}")

    dfs_1 = []
    for f in tsv_files:
       df = pd.read_csv(f, sep="\t", dtype={"chromosome": str})
       df["source_file"] = f.name
       dfs_1.append(df)

    df_long_1 = pd.concat(dfs_1, ignore_index=True)
    return df_long_1


def plot_ideogram_nucleosome_distance(df_long, suffix):

    # 1. PRÉPARATION DES DONNÉES
    df_long["chromosome"] = df_long["chromosome"].astype(str)
    df_long = df_long.sort_values(["chromosome", "start"])

    # 2. ORDRE DES CHROMOSOMES
    chroms = sorted([c for c in df_long["chromosome"].unique() if c.isdigit()], key=int)
    chroms += [c for c in ["X", "Y"] if c in df_long["chromosome"].unique()]

    n_chroms = len(chroms)

    # 3. COLORMAP ET NORMALISATION
    cmap = plt.cm.RdYlBu_r
    vmin = 7     # log2(128 bp)
    vmax = 10    # log2(1024 bp)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    nan_color = "#B9B9B9"

    # 4. FIGURE : un axe par chromosome
    fig, axes = plt.subplots(
        n_chroms, 1,
        figsize=(20, n_chroms * 0.55),
        gridspec_kw={"hspace": 0.25}
    )
    fig.subplots_adjust(left=0.08, top=0.97)  # ← rapproche titre et label Y

    if n_chroms == 1:
        axes = [axes]

    for ax, chrom in zip(axes, chroms):
        data = df_long[df_long["chromosome"] == chrom].sort_values("start")

        values = data["average_distance"].values

        masked = np.ma.masked_invalid(values)
        rgba = cmap(norm(masked.filled(fill_value=vmin)))
        rgba[np.isnan(values)] = mcolors.to_rgba(nan_color)

        img = rgba.reshape(1, len(values), 4)

        ax.imshow(
            img,
            aspect="auto",
            interpolation="nearest"
        )

        ax.set_yticks([0])
        ax.set_yticklabels([f"{chrom}"], fontsize=16, fontweight='bold', va="center")
        ax.set_xticks([])

        ax.set_facecolor(nan_color)

        for spine in ax.spines.values():
            spine.set_visible(False)

    # 5. COLORBAR GLOBALE + légende Y
    fig.text(0.05, 0.5, "Chromosomes", fontsize=24, fontweight='bold',
             va='center', ha='center', rotation='vertical')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                        fraction=0.01, pad=0.02, shrink=0.8)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("Distance moyenne entre nucléosomes (log2)", fontsize=22, fontweight='bold')

    # 6. LÉGENDE POUR LES NaN
    legend_elements = [
        Patch(facecolor="#2166AC", edgecolor="black", linewidth=0.5,
              label="Distance courte (<128 bp) — Nucléosomes serrés"),
        Patch(facecolor="#F7F7F7", edgecolor="black", linewidth=0.5,
              label="Distance normale (~200 bp) — Espacement canonique"),
        Patch(facecolor="#B2182B", edgecolor="black", linewidth=0.5,
              label="Distance longue (>1 kb) — Hétérochromatine"),
        Patch(facecolor=nan_color, edgecolor="black", linewidth=1,
              label="Données manquantes (NaN)")
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=2,
        fontsize=18,
        frameon=True,
        framealpha=0.95,
        title="Légende",
        title_fontsize=20
    )

    # 7. TITRE
    fig.suptitle(
        "Distribution spatiale de la distance moyenne des nucléosomes (log2)\n le long des chromosomes (bins de 10kb) dans cancer de la prostate",
        fontsize=16, fontweight="bold", y=1.00)

    # 8. SAUVEGARDER
    plt.savefig(
        f"ideogram_average_distance_{suffix}_prostate.svg",
        dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close()

    print(f"Saved ideogram_average_distance_{suffix}_prostate.svg")


#-------------
# 3. Main program
#-------------

if __name__ == "__main__":

    txt_files = [Path("/data2/jcolinge/fragmentomics/atlas/adal-prostate-wps-peaks/compiled-selection.txt"), Path("/data2/USERS/lruffinel/cfDNA/chromosomes_cover_uncover/compiled_Y_prostate.txt")]
    centromere_file = "/data2/USERS/lruffinel/cfDNA/chromosomes_cover_uncover/centromere-region.txt"
    

    tsv_dir = Path("/data2/USERS/lruffinel/cfDNA/chromosomes_cover_uncover/distance_nucleosome_norm_cancer/")
    tsv_dir.mkdir(exist_ok=True)

    # 1) Charger tous les TSV
    df_long = load_nucleosome_count_table(tsv_dir)
    df_long.to_csv("df_long_nucleosome_count_global_prostate.csv", index=False)

    # df_long_1 = load_average_distance_table(tsv_dir)
    # df_long_1.to_csv("df_long_average_distance_global_prostate.csv", index=False)

    # 2) Tracer les plots style idéogramme (haute résolution)
    plot_ideogram_nucleosome_count(df_long, "global", centromere_file=centromere_file)
    #plot_ideogram_nucleosome_distance(df_long_1, "global")
