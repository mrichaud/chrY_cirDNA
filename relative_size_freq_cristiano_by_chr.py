import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import chi2_contingency

# ============================
# PAR Definition & Helpers
# ============================
PAR = {
    "X": [
        (10_001 - 1, 2_781_479),         # PAR1 X
        (155_701_383 - 1, 156_030_895),  # PAR2 X
    ]
}

def norm_chrom(chrom):
    c = str(chrom).strip()
    if c.startswith("chr"):
        c = c[3:]
    return c

def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end

def is_in_par(chrom, start, end):
    c = norm_chrom(chrom)
    if c != "X":
        return False
    for p_start, p_end in PAR[c]:
        if overlaps(start, end, p_start, p_end):
            return True
    return False

def normalize_id_from_filename(path):
    base = os.path.basename(path)
    for suf in [
        ".hg38.frag.tsv",
        ".hg38.frag.bed",
        ".hg38.frag",
        ".frag.bed",
        ".frag",
        ".bed",
        ".tsv",
    ]:
        if base.endswith(suf):
            return base[:-len(suf)]
    return os.path.splitext(base)[0]


# =========================
# Histogram Processing
# =========================

def compute_histograms_for_target(indiv_tsv, target_chrom, L=1000, score_min=30, remove_par=True):
    """
    Computes size histograms for:
    1. target_chrom (Target chromosome to evaluate)
    2. background (All other chromosomes except target_chrom and Y)
    """
    df = pd.read_csv(
        indiv_tsv, sep="\t", header=None,
        names=["chr", "start", "end", "score", "strand"]
    )

    df = df[df["score"].astype(int) >= score_min]

    # Target pool: Autosomes + chrX
    valid_chroms = ["chr" + str(i) for i in range(1, 23)] + ["chrX"]
    df = df[df["chr"].isin(valid_chroms)]

    # PAR Region Removal on ChrX
    if remove_par:
        mask_par = (df["chr"] == "chrX") & df.apply(
            lambda r: is_in_par(r["chr"], int(r["start"]), int(r["end"])),
            axis=1
        )
        df = df[~mask_par]

    sizes = (df["end"] - df["start"]).to_numpy()
    chrs = df["chr"].to_numpy()

    mask = (sizes >= 0) & (sizes < L)
    sizes = sizes[mask].astype(int)
    chrs = chrs[mask]

    target_chr_str = "chr" + norm_chrom(target_chrom)
    
    # Split between target chromosome and the background pool
    sizes_target = sizes[chrs == target_chr_str]
    sizes_bg = sizes[chrs != target_chr_str]

    count_target = np.bincount(sizes_target, minlength=L).astype(np.int64)
    count_bg = np.bincount(sizes_bg, minlength=L).astype(np.int64)
    
    return count_target, count_bg


def process_curves_and_hists(name, target_chrom, files_or_pattern, L=1000, n_rep=1000, seed=0, score_min=30, remove_par=True):
    if isinstance(files_or_pattern, (list, tuple)):
        all_files = sorted(files_or_pattern)
    else:
        all_files = sorted(glob.glob(files_or_pattern))

    print(f"\n--- Cohort {name} | Target: {target_chrom} ({len(all_files)} files) ---")

    global_target = np.zeros(L, dtype=np.int64)
    global_bg = np.zeros(L, dtype=np.int64)

    for f in all_files:
        c_target, c_bg = compute_histograms_for_target(
            f, target_chrom=target_chrom, L=L, score_min=score_min, remove_par=remove_par
        )
        global_target += c_target
        global_bg += c_bg

    N_target = int(global_target.sum())
    if N_target == 0 or global_bg.sum() == 0:
        sizes = np.arange(L)
        curves = np.zeros((0, L), dtype=np.float32)
        return sizes, curves, global_target, global_bg, len(all_files)

    freq_target = (global_target / N_target).astype(np.float32)

    total_bg = float(global_bg.sum())
    p_bg = global_bg.astype(np.float64) / total_bg
    p_bg = np.nan_to_num(p_bg, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(p_bg, 0.0, 1.0, out=p_bg)

    s = p_bg.sum()
    if s <= 0:
        sizes = np.arange(L)
        curves = np.zeros((0, L), dtype=np.float32)
        return sizes, curves, global_target, global_bg, len(all_files)
    p_bg /= s

    rng = np.random.default_rng(seed)
    curves = np.empty((n_rep, L), dtype=np.float32)

    for b in range(n_rep):
        c_bg_boot = rng.multinomial(N_target, p_bg).astype(np.float32)
        freq_bg_boot = c_bg_boot / c_bg_boot.sum()
        curves[b] = freq_bg_boot - freq_target

    return np.arange(L), curves, global_target, global_bg, len(all_files)


def curve_band_from_curves(curves, alpha=0.20):
    curves = np.asarray(curves)
    if curves.shape[0] == 0:
        L = curves.shape[1] if curves.ndim == 2 else 0
        z = np.zeros(L, dtype=float)
        return z, z, z
    mean = curves.mean(axis=0)
    low = np.quantile(curves, alpha / 2, axis=0)
    high = np.quantile(curves, 1 - alpha / 2, axis=0)
    return mean, low, high


# =========================
# Plotting Helper
# =========================

def plot_superposed_diff(data, target_chrom, color_map, label_map, alpha_band=0.20, xlim=(50, 250), out_svg=None):
    plt.figure(figsize=(8, 6))

    for name, cohort_data in data.items():
        sizes = cohort_data["sizes"]
        curves = cohort_data["curves"]
        n_files = cohort_data["n_files"]

        c = color_map.get(name, "gray")
        lab = label_map.get(name, name)
        label = f"{lab} (n={n_files})"

        mean, low, high = curve_band_from_curves(curves, alpha=alpha_band)

        plt.fill_between(sizes, low, high, color=c, alpha=0.25, linewidth=0, zorder=1)
        plt.plot(sizes, mean, color=c, linewidth=2, alpha=0.95, zorder=2, label=label)

    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(167, color="orange", linewidth=3.5)
    plt.axvline(147, color="#C41CAD", linewidth=3.5)

    plt.title(f"Fragment Length Difference: Background vs Chr{target_chrom}", fontsize=14, fontweight='bold')
    plt.xlabel("Fragment length (bp)", fontsize=16, fontweight='bold')
    plt.ylabel("Frequency difference", fontsize=16, fontweight='bold')

    plt.xlim(*xlim)
    ax = plt.gca()

    major_ticks = np.arange(xlim[0], xlim[1] + 1, 25)
    ax.set_xticks(major_ticks) 
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_xticklabels([f"{t:d}" for t in major_ticks])

    ax.set_xticks([147, 167], minor=True)
    ax.xaxis.set_minor_formatter(mticker.FixedFormatter(["147 pb", "167 pb"]))
    ax.tick_params(axis="x", which="minor", length=6, pad=8, labelsize=15)

    minor_labels = ax.xaxis.get_minorticklabels()
    if len(minor_labels) >= 2:
        minor_labels[0].set_color("#C41CAD")  # 147
        minor_labels[0].set_fontweight("bold")
        minor_labels[1].set_color("orange")   # 167
        minor_labels[1].set_fontweight("bold")

    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_svg, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_svg}")


# ============================
# File Selection
# ============================
def select_files(annot, cancer_label, pattern):
    filtered = annot[(annot["Sexe"] == "M") & (annot["Cancer"] == cancer_label)]
    valid_ids = set(filtered["ID_FinaleDB"].dropna().astype(str))

    all_files = sorted(glob.glob(pattern))
    kept = []
    for f in all_files:
        file_id = normalize_id_from_filename(f)
        if file_id in valid_ids:
            kept.append(f)
    return kept


# =========================
# Main Execution Loop
# =========================

if __name__ == "__main__":

    os.environ["TMPDIR"] = "/data2/USERS/richaud/chrY/distribution_fragments/tmp_ks"
    output_dir = "./chrom_comparison_plots"
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    annot = pd.read_csv("/data2/USERS/lruffinel/cfDNA/distribution_fragments/cristiano/cristiano_patients_annotation.csv", sep=";")

    # colorectal = select_files(annot, cancer_label="Colorectal Cancer", pattern="/data2/USERS/richaud/finaledb/colorectal/cristiano/indivs/*.hg38.frag.bed")
    # pancreatic = select_files(annot, cancer_label="Pancreatic Cancer", pattern="/data2/USERS/richaud/finaledb/pancreatic/cristiano/indivs/*.hg38.frag.bed")
    healthy = select_files(annot, cancer_label="Healthy", pattern="/data2/USERS/BIOINFO/databases/FinaleDB/cristiano-data-2019/*.hg38.frag.tsv")
    lung_cancer = select_files(annot, cancer_label="Lung Cancer", pattern="/data2/USERS/richaud/finaledb/lung/cristiano/indivs/*.hg38.frag")

    inputs = {
        # "Colorectal_cancer": colorectal,
        # "Pancreatic_cancer": pancreatic,
        "Healthy": healthy,
        "Lung_cancer": lung_cancer,
    }

    color_map = {
        # "Colorectal_cancer": "blue",
        # "Pancreatic_cancer": "green",
        "Healthy": "#533623",
        "Lung_cancer": "#352770",
    }

    label_map = {
        # "Colorectal_cancer": "Colorectal cancer",
        # "Pancreatic_cancer": "Pancreatic cancer",
        "Healthy": "Healthy",
        "Lung_cancer": "Lung cancer",
    }

    # Parameters
    L = 1000
    N_REP = 1000
    SEED = 0
    SCORE_MIN = 30
    ALPHA_BAND = 0.20

    # List of all chromosomes except Y (1 to 22 + X)
    chromosomes_to_test = [str(i) for i in range(1, 23)] + ["X"]

    # Main Loop over Chromosomes
    for chrom in chromosomes_to_test:
        print(f"\n==========================================")
        print(f"   Processing Chromosome: chr{chrom}")
        print(f"==========================================")

        chrom_data = {}
        for name, pattern in inputs.items():
            sizes, curves, global_target, global_bg, n_files = process_curves_and_hists(
                name=name,
                target_chrom=chrom,
                files_or_pattern=pattern,
                L=L,
                n_rep=N_REP,
                seed=SEED,
                score_min=SCORE_MIN,
                remove_par=True,
            )
            chrom_data[name] = {
                "sizes": sizes,
                "curves": curves,
                "global_target": global_target,
                "global_bg": global_bg,
                "n_files": n_files,
            }

        out_svg = os.path.join(output_dir, f"relative_size_freq_diff_chr{chrom}.svg")
        plot_superposed_diff(
            chrom_data,
            target_chrom=chrom,
            color_map=color_map,
            label_map=label_map,
            alpha_band=ALPHA_BAND,
            xlim=(50, 250),
            out_svg=out_svg
        )