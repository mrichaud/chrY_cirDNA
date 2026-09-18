import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency
import matplotlib.ticker as mticker


# ============================
# Exclure PAR
# ============================
PAR = {
    "X": [
        (10_001 - 1, 2_781_479),          # PAR1 X: 10 001 - 2 781 479
        (155_701_383 - 1, 156_030_895),   # PAR2 X: 155 701 383 - 156 030 895
    ]
}

def norm_chrom(chrom):
    c = chrom.strip()
    if c.startswith("chr"):
        c = c[3:]
    return c  # ex: "X", "Y", "1", ...

def overlaps(a_start, a_end, b_start, b_end):
    # half-open overlap
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

def compute_histograms_from_tsv(indiv_tsv, L=1000, score_min=30, remove_par=True):
    df = pd.read_csv(
        indiv_tsv, sep="\t", header=None,
        names=["chr", "start", "end", "score", "strand"]
    )

    df = df[df["score"].astype(int) >= score_min]

    chr_autosomes = ["chr" + str(i) for i in range(1, 23)] + ["chrX"]
    df = df[df["chr"].isin(chr_autosomes + ["chrY"])]

    # ----------------------------
    # Exclusion des rÃ©gions PAR sur chrX
    # ----------------------------
    if remove_par:
        mask_par = (df["chr"] == "chrX") & df.apply(
            lambda r: is_in_par(r["chr"], int(r["start"]), int(r["end"])),
            axis=1
        )
        df = df[~mask_par]
    # ----------------------------

    sizes = (df["end"] - df["start"]).to_numpy()
    chrs = df["chr"].to_numpy()

    mask = (sizes >= 0) & (sizes < L)
    sizes = sizes[mask].astype(int)
    chrs = chrs[mask]

    sizes_Y = sizes[chrs == "chrY"]
    sizes_AX = sizes[chrs != "chrY"]

    count_Y = np.bincount(sizes_Y, minlength=L).astype(np.int64)
    count_AX = np.bincount(sizes_AX, minlength=L).astype(np.int64)
    return count_Y, count_AX


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

def process_curves_and_hists(name, files_or_pattern, L=1000, n_rep=1000, seed=0, score_min=30, remove_par=True):
 
    if isinstance(files_or_pattern, (list, tuple)):
        all_files = sorted(files_or_pattern)
    else:
        all_files = sorted(glob.glob(files_or_pattern))

    print(f"\n=== Cohorte Cristiano et Adalsteinsson {name} : {len(all_files)} fichiers ===")

    global_Y = np.zeros(L, dtype=np.int64)
    global_AX = np.zeros(L, dtype=np.int64)

    for f in all_files:
        cY, cAX = compute_histograms_from_tsv(f, L=L, score_min=score_min, remove_par=True)
        global_Y += cY
        global_AX += cAX

    N_Y = int(global_Y.sum())
    if N_Y == 0 or global_AX.sum() == 0:
        sizes = np.arange(L)
        curves = np.zeros((0, L), dtype=np.float32)
        return sizes, curves, global_Y, global_AX, len(all_files)

    # freqY en float32 
    freqY = (global_Y / global_Y.sum()).astype(np.float32)

    # pAX en float64 + nettoyage robuste
    total_AX = float(global_AX.sum())
    if total_AX <= 0:
        sizes = np.arange(L)
        curves = np.zeros((0, L), dtype=np.float32)
        return sizes, curves, global_Y, global_AX, len(all_files)

    pAX = global_AX.astype(np.float64) / total_AX
    pAX = np.nan_to_num(pAX, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(pAX, 0.0, 1.0, out=pAX)

    s = pAX.sum()
    if s <= 0:
        sizes = np.arange(L)
        curves = np.zeros((0, L), dtype=np.float32)
        return sizes, curves, global_Y, global_AX, len(all_files)
    pAX /= s

    rng = np.random.default_rng(seed)

    # curves en float32 (RAM)
    curves = np.empty((n_rep, L), dtype=np.float32)

    for b in range(n_rep):
        cAX_boot = rng.multinomial(N_Y, pAX)      # int64
        cAX_boot = cAX_boot.astype(np.float32)    # float32 aprÃ¨s tirage
        freqAX_boot = cAX_boot / cAX_boot.sum()
        curves[b] = freqAX_boot - freqY

    return np.arange(L), curves, global_Y, global_AX, len(all_files)


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


def band_label(alpha):
    if np.isclose(alpha, 0.10):
        return "Bande 5â€“95%"
    if np.isclose(alpha, 0.20):
        return "Bande 10â€“90%"
    if np.isclose(alpha, 0.50):
        return "IQR (25â€“75%)"
    return f"Bande (alpha={alpha})"


# =========================
#   Plot helpers
# =========================

def plot_superposed_diff(data, color_map, label_map, alpha_band=0.20, xlim=(50, 250), out_svg=None):
    plt.figure(figsize=(12, 6))

    line_handles = []
    band_handles = []
    legend_labels = []

    for name, data in data.items():
        sizes = data["sizes"]
        curves = data["curves"]
        n_files = data["n_files"]

        c = color_map.get(name, "gray")
        lab = label_map.get(name, name)
        label = f"{lab} (n={n_files})"

        mean, low, high = curve_band_from_curves(curves, alpha=alpha_band)

        plt.fill_between(sizes, low, high, color=c, alpha=0.25, linewidth=0, zorder=1)
        plt.plot(sizes, mean, color=c, linewidth=2, alpha=0.95, zorder=2)

        line_handles.append(Line2D([0], [0], color=c, lw=2))
        band_handles.append(Patch(facecolor=c, edgecolor="none", alpha=0.25))
        legend_labels.append(label)

    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(167, color="gold", linewidth=2)
    plt.axvline(147, color="#C41CAD", linewidth=2)


    plt.xlabel("Fragment length (bp)")
    plt.ylabel("Frequency difference")
    plt.title(
        "Superposition courbes de frÃ©quences relatives des donnÃ©es Cristiano et Adalsteisson (male) : freq(Autosomes+X) - freq(Y)\n"
        f"{band_label(alpha_band)} (bootstrap au niveau fragment; AX tirÃ© avec remise, taille = N_Y)"
    )
    plt.xlim(*xlim)
    ax = plt.gca()

    # 1) ticks MAJEURS identiques Ã  avant (ex: 50, 75, 100, ... 250)
    major_ticks = np.arange(xlim[0], xlim[1] + 1, 25)
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([f"{t:d}" for t in major_ticks])

    # 2) ticks MINEURS pour 147 et 167 + labels
    ax.set_xticks([147, 167], minor=True)
    ax.xaxis.set_minor_formatter(mticker.FixedFormatter(["147 pb", "167 pb"]))
    ax.tick_params(axis="x", which="minor", length=6, pad=8, labelsize=10)

    # colorer les 2 labels minor (optionnel)
    minor_labels = ax.xaxis.get_minorticklabels()
    if len(minor_labels) >= 2:
        minor_labels[0].set_color("#C41CAD")  # 147
        minor_labels[0].set_fontweight("bold")
        minor_labels[1].set_color("gold")     # 167
        minor_labels[1].set_fontweight("bold")


    handles, labels = [], []
    for lh, bh, lab in zip(line_handles, band_handles, legend_labels):
        handles.extend([lh, bh])
        labels.extend([lab, band_label(alpha_band)])

    plt.legend(handles, labels, frameon=False, title="Cancer", title_fontsize=10, fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(out_svg, dpi=300)
    plt.close()
    print(f"Figure gÃ©nÃ©rÃ©e : {out_svg}")


# ============================
# SÃ©lection des fichiers
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

    annot = pd.read_csv("/data2/USERS/lruffinel/cfDNA/distribution_fragments/cristiano/cristiano_patients_annotation.csv",sep=";")

    colorectal = select_files(annot, cancer_label="Colorectal Cancer", pattern="/data2/USERS/richaud/finaledb/colorectal/cristiano/indivs/*.hg38.frag.bed")
    pancreatic = select_files(annot, cancer_label="Pancreatic Cancer", pattern="/data2/USERS/richaud/finaledb/pancreatic/cristiano/indivs/*.hg38.frag.bed")
    healthy = select_files(annot, cancer_label="Healthy", pattern="/data2/USERS/BIOINFO/databases/FinaleDB/cristiano-data-2019/*.hg38.frag.tsv")
    lung_cancer = select_files(annot, cancer_label="Lung Cancer", pattern="/data2/USERS/richaud/finaledb/lung/cristiano/indivs/*.hg38.frag")
    prostates_cancer = glob.glob("/data2/USERS/richaud/finaledb/prostate/adalsteinsson/*.hg38.frag")

    inputs = {
        "Colorectal_cancer": colorectal,
        "Pancreatic_cancer": pancreatic,
        "Healthy": healthy,
        "Lung_cancer": lung_cancer,
        "Prostate_cancer": prostates_cancer,
    }

    color_map = {
        "Colorectal_cancer": "blue",
        "Pancreatic_cancer": "green",
        "Healthy": "#533623",
        "Lung_cancer": "#352770",
        "Prostate_cancer": "#F0691B",
    }
    

    label_map = {
        "Colorectal_cancer": "Colorectal cancer (cristiano)",
        "Pancreatic_cancer": "Pancreatic cancer (cristiano)",
        "Healthy": "Healthy (cristiano)",
        "Lung_cancer": "Lung cancer (cristiano)",
        "Prostate_cancer": "Prostate cancer (adalsteinsson)",
    }

    # ---- PARAMS
    L = 1000
    N_REP = 1000
    SEED = 0
    SCORE_MIN = 30

    # bande (pour Figure 1 et Figure 2)
    ALPHA_BAND = 0.20  # 10-90 par dÃ©faut; mets 0.50 pour IQR

    # FenÃªtres stats
    windows = [
        ("global_50_250", 50, 250),
        ("short_100_150", 100, 150),
        ("around167_150_180", 150, 180),
        ("longtail_180_250", 180, 250),
    ]

    # ---- 0) Calcul curves + hists une fois par dataset (pour rÃ©utiliser dans les diffÃ©rentes figures)
    data = {}
    for name, pattern in inputs.items():
        sizes, curves, global_Y, global_AX, n_files = process_curves_and_hists(
            name, pattern, L=L, n_rep=N_REP, seed=SEED, score_min=SCORE_MIN, remove_par=True,
        )
        data[name] = {
            "sizes": sizes,
            "curves": curves,
            "global_Y": global_Y,
            "global_AX": global_AX,
            "n_files": n_files,
        }

    # ---- Option A: Figure 1 = superposition des diff + bandes
    plot_superposed_diff(data, color_map=color_map, label_map=label_map, alpha_band=ALPHA_BAND, xlim=(50, 250), out_svg="relative_size_frequency_diff_superposed_cristiano_adalsteinsson.svg")
