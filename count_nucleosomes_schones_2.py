#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# ----------------
# ParamÃ¨tres
# ----------------
BED_DIR = Path("/data2/USERS/lruffinel/cfDNA/distribution_fragments/Shones/fragments_bed/")
PATTERN = "ActivatedNucleosomes-*.bed"

CHR_LENGTH_FILE = Path("/data2/USERS/lruffinel/cfDNA/chromosomes_cover_uncover/cover_tsv/pos_chromosomes.tsv")
CENTROMERE_FILE = Path("/data2/USERS/lruffinel/cfDNA/chromosomes_cover_uncover/centromere-region.txt")

OUT_DIR = Path("/data2/USERS/richaud/chrY/schones/")
OUT_TSV = OUT_DIR / "fragments_per_bin_by_chr_no_centromere.tsv"
OUT_PNG = OUT_DIR / "fragments_per_bin_by_chr_no_centromere_article.svg"

PAR_X = [
    (10_001 - 1, 2_781_479),
    (155_701_383 - 1, 156_030_895),
]

BIN_SIZE = 10_000  # 10kb

CHR_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


# ----------------
# Helpers
# ----------------
def extract_chr_from_filename(filename: str) -> Optional[str]:
    m = re.search(r"\b(chr(?:[1-9]|1[0-9]|2[0-2]|X|Y))\b", filename)
    return m.group(1) if m else None


def normalize_chr_token(token: str) -> Optional[str]:
    s = str(token).strip()
    if not s:
        return None
    if s.startswith("chr"):
        s2 = s
    else:
        s2 = "chr" + s
    if re.fullmatch(r"chr(?:[1-9]|1[0-9]|2[0-2]|X|Y)", s2):
        return s2
    return None


def load_chr_lengths(pos_chromosomes_tsv):
    df = pd.read_csv(
        pos_chromosomes_tsv,
        sep="\t",
        header=None,
        names=["idx", "chr", "taille", "start", "end"],
        dtype={"chr": str, "taille": "int64"},
    )
    df["chr"] = df["chr"].apply(normalize_chr_token)
    df = df.dropna(subset=["chr"])
    return df.groupby("chr")["taille"].max()


def load_centromere_regions(centromere_file):
    centromeres = {}
    
    with open(centromere_file) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            
            parts = line.split()
            chrom = parts[0]
            
            # VÃ©rifier que les coordonnÃ©es sont numÃ©riques
            if not all(p.isdigit() for p in parts[1:]):
                print(f"Ligne ignorÃ©e (non numÃ©rique) : {line.strip()}")
                continue
            
            coords = list(map(int, parts[1:]))
            
            # Si nombre impair, ignorer la derniÃ¨re coordonnÃ©e
            if len(coords) % 2 != 0:
                coords = coords[:-1]
            
            # Regrouper par paires (start, end)
            intervals = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            centromeres.setdefault(chrom, []).extend(intervals)
    
    return centromeres


def is_in_centromere(chrom, start, end, centromeres):
    if str(chrom) not in centromeres:
        return False
    
    for centro_start, centro_end in centromeres[str(chrom)]:
        if not (end <= centro_start or start >= centro_end):
            return True
    return False

def PAR_region(start, end, regions):
    for a, b in regions:
        if not (end <= a or start >= b):
            return True
    return False
# ----------------
# BED processing
# ----------------
def process_bed(bed_path, chrom, chr_lengths, centromeres, bin_size):
    n_total = 0
    n_kept = 0

    chrom_key = str(chrom).replace("chr", "")

    with bed_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("track") or line.startswith("browser"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue

            n_total += 1

            # 1) Filtrage centromÃ¨res (comme avant)
            if centromeres is not None and is_in_centromere(chrom_key, start, end, centromeres):
                continue

            # 2) Filtrage PAR sur X
            if chrom_key == "X" and PAR_region(start, end, PAR_X):
                continue

            n_kept += 1

    chr_length_total = int(chr_lengths.loc[chrom]) if (chr_lengths is not None and chrom in chr_lengths.index) else None

    excluded_length = 0
    if centromeres is not None and str(chrom_key) in centromeres:
        for centro_start, centro_end in centromeres[str(chrom_key)]:
            excluded_length += (centro_end - centro_start)

    # 3) Exclure aussi la longueur PAR dans la longueur effective
    if chrom_key == "X":
        for a, b in PAR_X:
            excluded_length += (b - a)

    chr_length_effective = (chr_length_total - excluded_length) if chr_length_total is not None else None

    nb_bins = int(math.ceil(chr_length_effective / bin_size)) if chr_length_effective and chr_length_effective > 0 else None
    fragments_per_bin = (n_kept / nb_bins) if nb_bins and nb_bins > 0 else None

    return {
        "chr": chrom,
        "chromosome": chrom_key,
        "file": bed_path.name,
        "bin_size": bin_size,
        "n_fragments_total": n_total,
        "n_fragments_non_centromere": n_kept,
        "pct_kept": (100.0 * n_kept / n_total) if n_total > 0 else None,
        "chr_length_total": chr_length_total,
        "excluded_length": excluded_length,
        "chr_length_effective": chr_length_effective,
        "nb_bins": nb_bins,
        "fragments_per_bin": fragments_per_bin,
    }

# ----------------
# Plot (style proche de ton exemple, sans IC95)
# ----------------
def comma_fmt(x, pos):
    # format y-axis en "1 234 567"
    try:
        return f"{int(x):,}".replace(",", " ")
    except Exception:
        return str(x)


def plot_barplot_fragments_per_bin(df_counts, out_svg):
    sns.set_theme(style="whitegrid")

    dfp = df_counts.copy()
    dfp["chromosome"] = dfp["chromosome"].astype(str)

    chrom_order = [str(i) for i in range(1, 23)] + ["X", "Y"]
    dfp = dfp[dfp["chromosome"].isin(chrom_order)].copy()
    dfp["chromosome"] = pd.Categorical(dfp["chromosome"], categories=chrom_order, ordered=True)
    dfp = dfp.sort_values("chromosome")


    # couleurs autosomes vs sex
    colors = ["#5975A4" if c not in ["X", "Y"] else "#5F9E6E" for c in dfp["chromosome"].astype(str).tolist()]

    fig, ax = plt.subplots(figsize=(11, 8))

    x_pos = np.arange(len(dfp))
    heights = dfp["fragments_per_bin"].values

    ax.bar(x_pos, heights, color=colors, edgecolor="black", linewidth=0.8)
    ax.tick_params(labelsize=20)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(dfp["chromosome"].astype(str).values, rotation=0, fontsize=18)

    ax.set_xlabel("Chromosomes", fontsize=20, fontweight="bold")
    ax.set_ylabel("Nombre de fragments (bins de 10kb)", fontsize=20, fontweight="bold")
    

    ax.yaxis.set_major_formatter(FuncFormatter(comma_fmt))
    ax.grid(axis="y", color="lightgray", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_svg, dpi=300, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close()

    print(f"Plot saved: {out_svg}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(BED_DIR.glob(PATTERN))
    if not files:
        raise SystemExit(f"Aucun fichier trouvÃ©: {BED_DIR}/{PATTERN}")

    chr_lengths = load_chr_lengths(CHR_LENGTH_FILE)
    print(f"[lengths] {len(chr_lengths)} chromosomes chargÃ©s: {CHR_LENGTH_FILE}")

    centromeres = load_centromere_regions(CENTROMERE_FILE)
    print(f"[centromeres] {len(centromeres)} chromosomes chargÃ©s: {CENTROMERE_FILE}")

    rows = []
    total_files = len(files)

    for i, fp in enumerate(files, start=1):
        chrom = extract_chr_from_filename(fp.name)
        if chrom is None:
            print(f"[{i}/{total_files}] SKIP (chr non reconnu): {fp.name}", flush=True)
            continue

        print(f"[{i}/{total_files}] Processing {chrom} ({fp.name}) ...", flush=True)
        row = process_bed(
            bed_path=fp,
            chrom=chrom,
            chr_lengths=chr_lengths,
            centromeres=centromeres,
            bin_size=BIN_SIZE,
        )
        print(
            f"[{i}/{total_files}] Done {chrom}: "
            f"total={row['n_fragments_total']:,} kept={row['n_fragments_non_centromere']:,} "
            f"bins={row['nb_bins']} per_bin={row['fragments_per_bin']}",
            flush=True,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Aucun rÃ©sultat produit (check extract_chr).")

    # ordre stable
    df["chr"] = pd.Categorical(df["chr"], categories=CHR_ORDER, ordered=True)
    df = df.sort_values("chr")

    # on Ã©crit le TSV complet (avec bins, longueurs, etc.)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"[OK] TSV Ã©crit: {OUT_TSV}")

    # plot UNIQUEMENT fragments_per_bin (pas de counts bruts)
    plot_barplot_fragments_per_bin(df, OUT_PNG)

    print("terminÃ©")
