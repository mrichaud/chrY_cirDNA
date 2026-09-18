import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.ticker as mticker

# ============================
# Exclure PAR
# ============================
PAR = {
    "X": [
        (10_001 - 1, 2_781_479),         # PAR1 X: 10 001 - 2 781 479
        (155_701_383 - 1, 156_030_895),  # PAR2 X: 155 701 383 - 156 030 895
    ]
}

def norm_chrom(chrom):
    c = chrom.strip()
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
            return base[: -len(suf)]
    return os.path.splitext(base)[0]

# =========================
#   Utils: histos / windows
# =========================

def compute_histograms_from_tsv(indiv_tsv, target_chr="chr17", L=1000, score_min=30, remove_par=True):
    """
    Computes size histograms for target_chr vs all other valid chromosomes (chr1-22, chrX, chrY).
    """
    target_chr = target_chr if target_chr.startswith("chr") else "chr" + target_chr
    
    df = pd.read_csv(
        indiv_tsv, sep="\t", header=None,
        names=["chr", "start", "end", "score", "strand"]
    )

    df = df[df["score"].astype(int) >= score_min]

    valid_chrs = ["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"]
    df = df[df["chr"].isin(valid_chrs)]

    # ----------------------------
    # Exclusion des regions PAR sur chrX
    # ----------------------------
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

    sizes_target = sizes[chrs == target_chr]
    sizes_rest = sizes[chrs != target_chr]

    count_target = np.bincount(sizes_target, minlength=L).astype(np.int64)
    count_rest = np.bincount(sizes_rest, minlength=L).astype(np.int64)
    return count_target, count_rest


def _slice_sum(hist, lo, hi):
    lo = max(0, int(lo))
    hi = min(len(hist) - 1, int(hi))
    if hi < lo:
        return 0
    return int(hist[lo:hi + 1].sum())


def chi2_window_in_vs_out(countA, countB, lo, hi):
    A_in = _slice_sum(countA, lo, hi)
    B_in = _slice_sum(countB, lo, hi)
    A_tot = int(countA.sum())
    B_tot = int(countB.sum())
    A_out = A_tot - A_in
    B_out = B_tot - B_in

    if A_tot == 0 or B_tot == 0:
        return np.nan, np.nan

    table = np.array([[A_in, A_out], [B_in, B_out]], dtype=float)
    chi2, p, dof, expected = chi2_contingency(table, correction=False)
    return float(chi2), float(p)


# =========================
#   Bootstrap curves (diff)
# =========================

def process_curves_and_hists(name, files_or_pattern, target_chr="chr17", L=1000, n_rep=1000, seed=0, score_min=30, remove_par=True):
    if isinstance(files_or_pattern, (list, tuple)):
        all_files = sorted(files_or_pattern)
    else:
        all_files = sorted(glob.glob(files_or_pattern))

    print(f"\n=== Cohorte Cristiano {name} ({target_chr} vs Rest) : {len(all_files)} fichiers ===")

    global_target = np.zeros(L, dtype=np.int64)
    global_rest = np.zeros(L, dtype=np.int64)

    for f in all_files:
        c_target, c_rest = compute_histograms_from_tsv(
            f, target_chr=target_chr, L=L, score_min=score_min, remove_par=remove_par
        )
        global_target += c_target
        global_rest += c_rest

    N_target = int(global_target.sum())
    if N_target == 0 or global_rest.sum() == 0:
        sizes = np.arange(L)
        curves = np.zeros((0, L), dtype=np.float32)
        return sizes, curves, global_target, global_rest, len(all_files)

    # Relative frequencies: Target vs Rest
    freq_target = (global_target / global_target.sum()).astype(np.float32)

    total_rest = float(global_rest.sum())
    if total_rest <= 0:
        sizes = np.arange(L)
        curves = np.zeros((0, L), dtype=np.float32)
        return sizes, curves, global_target, global_rest, len(all_files)

    p_rest = global_rest.astype(np.float64) / total_rest
    p_rest = np.nan_to_num(p_rest, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(p_rest, 0.0, 1.0, out=p_rest)

    s = p_rest.sum()
    if s <= 0:
        sizes = np.arange(L)
        curves = np.zeros((0, L), dtype=np.float32)
        return sizes, curves, global_target, global_rest, len(all_files)
    p_rest /= s

    rng = np.random.default_rng(seed)
    curves = np.empty((n_rep, L), dtype=np.float32)

    # Multinomial resampling matching the depth of the target chromosome
    for b in range(n_rep):
        c_rest_boot = rng.multinomial(N_target, p_rest).astype(np.float32)
        freq_rest_boot = c_rest_boot / c_rest_boot.sum()
        curves[b] = freq_rest_boot - freq_target

    return np.arange(L), curves, global_target, global_rest, len(all_files)


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
#   Plot helpers
# =========================

def plot_superposed_diff(data, color_map, label_map, target_chr="chr17", alpha_band=0.20, xlim=(50, 250), out_svg=None):
    plt.figure(figsize=(8, 6))

    for name, cohort_data in data.items():
        sizes = cohort_data["sizes"]
        curves = cohort_data["curves"]
        n_files = cohort_data["n_files"]

        c = color_map.get(name, "gray")
        lab = label_map.get(name, name)

        mean, low, high = curve_band_from_curves(curves, alpha=alpha_band)

        plt.fill_between(sizes, low, high, color=c, alpha=0.25, linewidth=0, zorder=1)
        plt.plot(sizes, mean, color=c, linewidth=2, alpha=0.95, zorder=2, label=f"{lab} (n={n_files})")

    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(167, color="orange", linewidth=3.5)
    plt.axvline(147, color="#C41CAD", linewidth=3.5)

    plt.xlabel("Fragment length (bp)", fontsize=16, fontweight='bold')
    plt.ylabel(f"Frequency diff (Rest - {target_chr})", fontsize=16, fontweight='bold')
    plt.title(f"{target_chr} vs Rest of Genome", fontsize=16, fontweight='bold')

    plt.xlim(*xlim)
    ax = plt.gca()

    # Major ticks
    major_ticks = np.arange(xlim[0], xlim[1] + 1, 25)
    ax.set_xticks(major_ticks)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_xticklabels([f"{t:d}" for t in major_ticks])

    # Minor ticks at 147 and 167 bp
    ax.set_xticks([147, 167], minor=True)
    ax.xaxis.set_minor_formatter(mticker.FixedFormatter(["147 pb", "167 pb"]))
    ax.tick_params(axis="x", which="minor", length=6, pad=8, labelsize=15)

    minor_labels = ax.xaxis.get_minorticklabels()
    if len(minor_labels) >= 2:
        minor_labels[0].set_color("#C41CAD")  # 147
        minor_labels[0].set_fontweight("bold")
        minor_labels[1].set_color("orange")   # 167
        minor_labels[1].set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(out_svg, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure generated : {out_svg}")


# ============================
# Selection des fichiers
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
#   Main
# =========================

if __name__ == "__main__":

    os.environ["TMPDIR"] = "/data2/USERS/richaud/chrY/distribution_fragments/tmp_ks"
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)

    annot = pd.read_csv("/data2/USERS/lruffinel/cfDNA/distribution_fragments/cristiano/cristiano_patients_annotation.csv", sep=";")

    colorectal = select_files(annot, cancer_label="Colorectal Cancer", pattern="/data2/USERS/richaud/finaledb/colorectal/cristiano/indivs/*.hg38.frag.bed")
    pancreatic = select_files(annot, cancer_label="Pancreatic Cancer", pattern="/data2/USERS/richaud/finaledb/pancreatic/cristiano/indivs/*.hg38.frag.bed")
    healthy = select_files(annot, cancer_label="Healthy", pattern="/data2/USERS/BIOINFO/databases/FinaleDB/cristiano-data-2019/*.hg38.frag.tsv")
    lung_cancer = select_files(annot, cancer_label="Lung Cancer", pattern="/data2/USERS/richaud/finaledb/lung/cristiano/indivs/*.hg38.frag")

    inputs = {
        "Colorectal_cancer": colorectal,
        "Pancreatic_cancer": pancreatic,
        "Healthy": healthy,
        "Lung_cancer": lung_cancer,
    }

    color_map = {
        "Colorectal_cancer": "blue",
        "Pancreatic_cancer": "green",
        "Healthy": "#533623",
        "Lung_cancer": "#352770",
    }

    label_map = {
        "Colorectal_cancer": "Colorectal cancer (cristiano)",
        "Pancreatic_cancer": "Pancreatic cancer (cristiano)",
        "Healthy": "Healthy (cristiano)",
        "Lung_cancer": "Lung cancer (cristiano)",
    }

    # ---- PARAMS
    L = 1000
    N_REP = 1000
    SEED = 0
    SCORE_MIN = 30
    ALPHA_BAND = 0.20  # 10-90 quantile band

    # List of target chromosomes to iterate over
    target_chromosomes = ["chr22"]

    for target_chr in target_chromosomes:
        print(f"\n==========================================")
        print(f" PROCESSING: {target_chr} vs Rest of Genome")
        print(f"==========================================")
        
        data = {}
        for name, file_list in inputs.items():
            sizes, curves, global_target, global_rest, n_files = process_curves_and_hists(
                name=name,
                files_or_pattern=file_list,
                target_chr=target_chr,
                L=L,
                n_rep=N_REP,
                seed=SEED,
                score_min=SCORE_MIN,
                remove_par=True
            )
            data[name] = {
                "sizes": sizes,
                "curves": curves,
                "global_target": global_target,
                "global_rest": global_rest,
                "n_files": n_files,
            }

        # Export individual SVGs per chromosome
        out_svg = f"relative_size_frequency_diff_superposed_cristiano_{target_chr}.svg"
        plot_superposed_diff(
            data=data,
            color_map=color_map,
            label_map=label_map,
            target_chr=target_chr,
            alpha_band=ALPHA_BAND,
            xlim=(50, 250),
            out_svg=out_svg
        )