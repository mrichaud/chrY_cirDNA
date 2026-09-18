#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMÈTRES (PAR)
# ═══════════════════════════════════════════════════════════════════════════════
PAR_X = [
    (10_001 - 1, 2_781_479),
    (155_701_383 - 1, 156_030_895),
]
NUC_LEN = 147  # longueur utilisée pour start/end lors du filtrage


def PAR_region(start, end, regions):
    """True si [start,end) chevauche un intervalle de regions."""
    for a, b in regions:
        if not (end <= a or start >= b):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# PARTIE 1 : CHARGEMENT ET PRÉPARATION DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════
def chromosome_length(tsv_nucleosome):
    nucleosomes = pd.read_csv(
        tsv_nucleosome,
        sep="\t",
        header=None,
        names=["idx", "chr", "taille", "start", "end"],
        dtype=str,
    )
    nucleosomes["chromosome"] = nucleosomes["chr"].str.replace("chr", "", regex=False)
    nucleosomes["taille"] = nucleosomes["taille"].astype(int)
    chr_lengths = nucleosomes.groupby("chromosome")["taille"].max()
    return chr_lengths


def load_centromere_regions(centromere_file):
    centromeres = {}

    with open(centromere_file) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue

            parts = line.split()
            if not parts:
                continue

            # ignore header du type: chr centro_start centro_end
            if parts[0].lower() == "chr":
                continue

            chrom = parts[0]

            # Vérifier que les coordonnées sont numériques
            if not all(p.isdigit() for p in parts[1:]):
                print(f"Ligne ignorée (non numérique) : {line.strip()}")
                continue

            coords = list(map(int, parts[1:]))

            # Si nombre impair, ignorer la dernière coordonnée
            if len(coords) % 2 != 0:
                coords = coords[:-1]

            # Regrouper par paires (start, end)
            intervals = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            centromeres.setdefault(chrom, []).extend(intervals)

    return centromeres


def is_in_centromere(chrom, start, end, centromeres):
    if str(chrom) not in centromeres:
        return False

    for centro_start, centro_end in centromeres[str(chrom)]:
        if not (end <= centro_start or start >= centro_end):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORISATION : filtrage + longueurs effectives (centromères + PAR_X)
# ═══════════════════════════════════════════════════════════════════════════════
def is_excluded_region(chrom, start, end, centromeres):
  
    if centromeres is not None and is_in_centromere(chrom, start, end, centromeres):
        return True
    if str(chrom) == "X" and PAR_region(start, end, PAR_X):
        return True
    return False


def excluded_length_bp(chrom, centromeres):
 
    excluded = 0

    if centromeres is not None and str(chrom) in centromeres:
        for a, b in centromeres[str(chrom)]:
            excluded += (b - a)

    if str(chrom) == "X":
        for a, b in PAR_X:
            excluded += (b - a)

    return excluded


def filter_excluded_regions(nucleosomes: pd.DataFrame, centromeres, nuc_len: int = NUC_LEN) -> pd.DataFrame:
   
    keep_mask = nucleosomes.apply(
        lambda r: is_excluded_region(
            r["chromosome"],
            r["position"],
            r["position"] + nuc_len,
            centromeres,
        ),
        axis=1,
    ).eq(False)

    return nucleosomes.loc[keep_mask].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTIE 2 : CALCULS STATISTIQUES
# ═══════════════════════════════════════════════════════════════════════════════
def count_nucleosomes_by_chromosome(txt_file, output_file, chr_lengths=None, centromeres=None, bin_size=10000):
    
    print(f"Chargement de {txt_file}...")
    nucleosomes = pd.read_csv(txt_file, sep="\t")

    if "chr" in nucleosomes.columns and "position" in nucleosomes.columns:
        nucleosomes = nucleosomes.rename(columns={"chr": "chromosome"})
    else:
        raise ValueError(f"Colonnes attendues : 'chr' et 'position'. Trouvées : {nucleosomes.columns.tolist()}")

    nucleosomes["chromosome"] = nucleosomes["chromosome"].astype(str).str.replace("chr", "", regex=False)
    nucleosomes["position"] = nucleosomes["position"].astype(int)

    print(f"Nucléosomes avant filtrage : {len(nucleosomes):,}")

    # --- Filtrage centromères + PAR_X ---
    n0 = len(nucleosomes)
    if centromeres is not None:
        nucleosomes = filter_excluded_regions(nucleosomes, centromeres, nuc_len=NUC_LEN)
    else:
        # si pas de centromères, on filtre quand même PAR_X
        nucleosomes = filter_excluded_regions(nucleosomes, centromeres=None, nuc_len=NUC_LEN)

    print(f"Nucléosomes après filtrage (centromères + PAR_X) : {len(nucleosomes):,} ({100 * len(nucleosomes) / n0:.2f}%)")

    # --- Longueur effective (factorisée) ---
    def get_effective_length(chrom, total_length=None):
        if total_length is None:
            if chr_lengths is not None and chrom in chr_lengths.index:
                total_length = chr_lengths[chrom]
            else:
                return None
        return total_length - excluded_length_bp(chrom, centromeres)

    # --- Comptage par chromosome ---
    counts = nucleosomes.groupby("chromosome").size().reset_index(name="nucleosome_count_effective")

    # --- Ajout des longueurs ---
    if chr_lengths is not None:
        counts["chr_length_total"] = counts["chromosome"].map(chr_lengths)
        counts["chr_length_effective"] = counts.apply(
            lambda row: get_effective_length(row["chromosome"], row["chr_length_total"]),
            axis=1,
        )
    else:
        max_positions = nucleosomes.groupby("chromosome")["position"].max()
        counts["chr_length_total"] = counts["chromosome"].map(max_positions)
        counts["chr_length_effective"] = counts["chr_length_total"]

    # --- Normalisation par longueur effective (par Mb) ---
    counts["chr_length_effective_normalized"] = (
        counts["nucleosome_count_effective"] / counts["chr_length_effective"]
    ) * 1_000_000

    # --- Pourcentage du total ---
    total_nucleosomes = counts["nucleosome_count_effective"].sum()
    counts["pct_of_total"] = (counts["nucleosome_count_effective"] / total_nucleosomes) * 100

    # --- Pourcentage de longueur filtrée ---
    counts["pct_length_filtered"] = (
        (counts["chr_length_total"] - counts["chr_length_effective"]) / counts["chr_length_total"] * 100
    )

    # --- Normalisation par bins ---
    counts["nb_bins"] = (counts["chr_length_effective"] / bin_size).astype(int)
    counts["nucleosomes_per_bin"] = counts["nucleosome_count_effective"] / counts["nb_bins"]

    # --- Calcul IC95% ---
    def calculate_se_and_ci(row, nucleosomes_df, bin_size):
        chrom = row["chromosome"]
        chrom_nucs = nucleosomes_df[nucleosomes_df["chromosome"] == chrom].copy()

        if len(chrom_nucs) == 0:
            return pd.Series({"SE": 0, "IC95_lower": 0, "IC95_upper": 0})

        chr_length = row["chr_length_effective"]
        bins = np.arange(0, chr_length + bin_size, bin_size)

        chrom_nucs["bin"] = pd.cut(chrom_nucs["position"], bins=bins, labels=False, include_lowest=True)
        bin_counts = chrom_nucs.groupby("bin").size()

        all_bins = pd.Series(0, index=range(len(bins) - 1))
        all_bins.update(bin_counts)

        se = all_bins.std() / np.sqrt(len(all_bins))
        mean_count = all_bins.mean()
        ic95_lower = max(0, mean_count - 1.96 * se)
        ic95_upper = mean_count + 1.96 * se

        return pd.Series({"SE": se, "IC95_lower": ic95_lower, "IC95_upper": ic95_upper})

    ci_data = counts.apply(lambda row: calculate_se_and_ci(row, nucleosomes, bin_size), axis=1)
    counts = pd.concat([counts, ci_data], axis=1)

    # --- Catégorie chromosome ---
    def categorize_chromosome(chrom):
        if str(chrom) in ["X", "Y"]:
            return "Sex"
        elif str(chrom) in ["M", "MT"]:
            return "Mitochondrial"
        else:
            return "Autosome"

    counts["chromosome_type"] = counts["chromosome"].apply(categorize_chromosome)

    # --- Tri des chromosomes ---
    def chromosome_sort_key(chrom):
        chrom_str = str(chrom)
        try:
            return (0, int(chrom_str), "")
        except ValueError:
            pass
        special_order = {"X": (1, 0, "X"), "Y": (1, 1, "Y")}
        if chrom_str in special_order:
            return special_order[chrom_str]
        return (2, 0, chrom_str)

    counts["_sort_key"] = counts["chromosome"].apply(chromosome_sort_key)
    counts = counts.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)

    counts.to_csv(output_file, sep="\t", index=False)
    print(f"Saved {output_file}")

    return counts


def calculate_average_distance_by_chromosome(txt_file, output_file, chr_lengths=None, centromeres=None):

    print(f"Chargement de {txt_file}...")
    nucleosomes = pd.read_csv(txt_file, sep="\t")

    if "chr" in nucleosomes.columns:
        nucleosomes = nucleosomes.rename(columns={"chr": "chromosome"})

    nucleosomes["chromosome"] = nucleosomes["chromosome"].astype(str).str.replace("chr", "", regex=False)
    nucleosomes["position"] = nucleosomes["position"].astype(int)

    # --- Filtrage centromères + PAR_X ---
    n0 = len(nucleosomes)
    nucleosomes = filter_excluded_regions(nucleosomes, centromeres, nuc_len=NUC_LEN)
    print(f"Nucléosomes après filtrage (centromères + PAR_X) : {len(nucleosomes):,} ({100 * len(nucleosomes) / n0:.2f}%)")

    def get_effective_length(chrom, total_length):
        return total_length - excluded_length_bp(chrom, centromeres)

    results = []
    for chrom in sorted(nucleosomes["chromosome"].unique()):
        chrom_df = nucleosomes[nucleosomes["chromosome"] == chrom].sort_values("position")
        chrom_df["distance"] = chrom_df["position"].diff()

        avg_distance = chrom_df["distance"].mean()
        median_distance = chrom_df["distance"].median()
        std_distance = chrom_df["distance"].std()
        nucleosome_count = len(chrom_df)

        if chr_lengths is not None and chrom in chr_lengths.index:
            chr_length_total = chr_lengths[chrom]
            chr_length_effective = get_effective_length(chrom, chr_length_total)
        else:
            chr_length_total = chrom_df["position"].max()
            chr_length_effective = chr_length_total

        results.append(
            {
                "chromosome": chrom,
                "chr_length_total": chr_length_total,
                "chr_length_effective": chr_length_effective,
                "average_distance": avg_distance,
                "median_distance": median_distance,
                "std_distance": std_distance,
                "nucleosome_count": nucleosome_count,
            }
        )

    result_df = pd.DataFrame(results)

    def categorize_chromosome(chrom):
        if str(chrom) in ["X", "Y"]:
            return "Sex"
        elif str(chrom) in ["M", "MT"]:
            return "Mitochondrial"
        else:
            return "Autosome"

    result_df["chromosome_type"] = result_df["chromosome"].apply(categorize_chromosome)

    result_df["SE"] = result_df["std_distance"] / np.sqrt(result_df["nucleosome_count"])
    result_df["IC95_lower"] = result_df["average_distance"] - 1.96 * result_df["SE"]
    result_df["IC95_upper"] = result_df["average_distance"] + 1.96 * result_df["SE"]

    result_df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved {output_file}")

    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# PARTIE 3 : VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════
def plot_barplot_nucleosome_count(df_counts, output_file):
    sns.set_theme(style="whitegrid")

    chrom_order = sorted(
        [c for c in df_counts["chromosome"].unique() if c not in ["X", "Y"]],
        key=lambda x: int(x) if str(x).isdigit() else 99,
    ) + ["X", "Y"]

    df_counts = df_counts[df_counts["chromosome"].isin(chrom_order)].copy()
    df_counts["chromosome"] = pd.Categorical(df_counts["chromosome"], categories=chrom_order, ordered=True)
    df_counts = df_counts.sort_values("chromosome")

    colors = ["#5975A4" if c not in ["X", "Y"] else "#5F9E6E" for c in chrom_order]

    fig, ax = plt.subplots(figsize=(14, 8))

    x_pos = np.arange(len(df_counts))
    heights = df_counts["nucleosomes_per_bin"].values

    ax.bar(x_pos, heights, color=colors, edgecolor="black", linewidth=0.8)
    

    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_counts["chromosome"].values, rotation=0)
    ax.set_xlabel("Chromosomes", fontsize=12, fontweight="bold")
    ax.set_ylabel("Nombre de nucléosomes (bins de 10kb)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Nombre de nucléosomes par chromosome (sans régions centromériques + sans PAR_X)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.grid(axis="y", color="lightgray", linestyle="--", linewidth=0.5, alpha=0.7)

    legend_elements = [
        Patch(facecolor="#5975A4", edgecolor="black", label="Autosome"),
        Patch(facecolor="#5F9E6E", edgecolor="black", label="Sex"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.05, 1.1),
        fontsize=7,
        frameon=True,
        shadow=True,
        fancybox=True,
        framealpha=0.95,
        title="Légende",
        title_fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_file}")


def plot_barplot_average_distance(df_distances, output_file):
    sns.set_theme(style="whitegrid")

    chrom_order = sorted(
        [c for c in df_distances["chromosome"].unique() if c not in ["X", "Y"]],
        key=lambda x: int(x) if str(x).isdigit() else 99,
    ) + ["X", "Y"]

    df_distances = df_distances[df_distances["chromosome"].isin(chrom_order)].copy()
    df_distances["chromosome"] = pd.Categorical(df_distances["chromosome"], categories=chrom_order, ordered=True)
    df_distances = df_distances.sort_values("chromosome")

    colors = ["#5975A4" if c not in ["X", "Y"] else "#5F9E6E" for c in chrom_order]

    fig, ax = plt.subplots(figsize=(14, 8))

    x_pos = np.arange(len(df_distances))
    heights = df_distances["average_distance"].values

    ax.bar(x_pos, heights, color=colors, edgecolor="black", linewidth=0.8)
    

    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_distances["chromosome"].values, rotation=0)
    ax.set_xlabel("Chromosomes", fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance moyenne entre nucléosomes (pb)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Distance moyenne entre nucléosomes par chromosome (sans régions centromériques + sans PAR_X)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.grid(axis="y", color="lightgray", linestyle="--", linewidth=0.5, alpha=0.7)

    legend_elements = [
        Patch(facecolor="#5975A4", edgecolor="black", label="Autosome"),
        Patch(facecolor="#5F9E6E", edgecolor="black", label="Sex"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.05, 1.1),
        fontsize=7,
        frameon=True,
        shadow=True,
        fancybox=True,
        framealpha=0.95,
        title="Légende",
        title_fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_file}")


# ═══════════════════════════════════════════════════════════════════════════════
# PARTIE 4 : PROGRAMME PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    nucleosome_file = Path("/data2/jcolinge/fragmentomics/atlas/cris-healthy-wps-peaks/compiled-selection-with-Y.txt")
    chr_length_file = Path("/data2/USERS/lruffinel/cfDNA/chromosomes_cover_uncover/cover_tsv/pos_chromosomes.tsv")
    centromere_file = Path("/data2/USERS/lruffinel/cfDNA/chromosomes_cover_uncover/centromere-region.txt")

    output_dir = Path("/data2/USERS/richaud/chrY")
    output_dir.mkdir(exist_ok=True)

    print("ÉTAPE 1 : Chargement des données de référence")

    chr_lengths = chromosome_length(chr_length_file)
    print(f"Longueurs de chromosomes chargées : {len(chr_lengths)} chromosomes")

    centromeres = load_centromere_regions(centromere_file)
    print(f"Régions centromériques chargées : {len(centromeres)} chromosomes")

    print("ÉTAPE 2 : Calculs statistiques")

    output_count = output_dir / "nucleosome_counts_global_sans_PAR.tsv"
    df_counts = count_nucleosomes_by_chromosome(
        txt_file=nucleosome_file,
        output_file=output_count,
        chr_lengths=chr_lengths,
        centromeres=centromeres,
        bin_size=10000,
    )

    output_distance = output_dir / "average_distance_global_sans_PAR.tsv"
    df_distances = calculate_average_distance_by_chromosome(
        txt_file=nucleosome_file,
        output_file=output_distance,
        chr_lengths=chr_lengths,
        centromeres=centromeres,
    )

    print("ÉTAPE 3 : Génération des graphiques")

    plot_barplot_nucleosome_count(
        df_counts,
        output_file=output_dir / "barplot_nucleosome_counts_IC95_global_normalisee_sans_PAR.svg",
    )

    plot_barplot_average_distance(
        df_distances,
        output_file=output_dir / "barplot_average_distance_IC95_global_normalisee_sans_PAR.svg",
    )

    print("PIPELINE TERMINÉ AVEC SUCCÈS")