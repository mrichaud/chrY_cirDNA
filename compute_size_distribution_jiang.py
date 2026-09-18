import pandas as pd
import numpy as np
import glob
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt


# ============================
# Exclure PAR
# ============================
PAR = {
    "X": [
        (10_001 - 1, 2_781_479),
        (155_701_383 - 1, 156_030_895),
    ]
}

def norm_chrom(chrom: str) -> str:
    c = chrom.strip()
    if c.startswith("chr"):
        c = c[3:]
    return c

def overlaps(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end

def is_in_par(chrom: str, start: int, end: int) -> bool:
    c = norm_chrom(chrom)
    if c != "X":
        return False
    for p_start, p_end in PAR[c]:
        if overlaps(start, end, p_start, p_end):
            return True
    return False

# ============================
# Helpers
# ============================
def load_profile(tsv_path):
    arr = np.loadtxt(tsv_path)
    if arr.shape[0] != 1000:
        raise ValueError(f"Taille inattendue: {tsv_path} -> {arr.shape}")
    return arr

def frac_in_window(profile, start_bp, end_bp):
    return float(np.sum(profile[start_bp:end_bp]))

def sample_id_from_path(path):
    return os.path.basename(path).split("_chr")[0]

def load_matrix(files):
    mats = []
    for f in files:
        try:
            mats.append(load_profile(f))
        except Exception as e:
            print(f"Skip {f} : {e}")
    return np.array(mats)

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

# ============================
# Wilcoxon helpers
# ============================
def wilcoxon_paired(a, b) -> float:
    """
    Wilcoxon signed-rank test appariÃ© (bilatÃ©ral).
    Retourne la p-value (float) ou np.nan si non calculable.
    """
    a = pd.Series(a).astype(float)
    b = pd.Series(b).astype(float)

    mask = ~(a.isna() | b.isna())
    a = a[mask].values
    b = b[mask].values

    if len(a) < 3:
        return np.nan

    if np.allclose(a - b, 0):
        return 1.0

    try:
        stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(p)
    except Exception:
        return np.nan

def format_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p=NA"
    if p < 1e-4:
        return "p<1e-4"
    return f"p={p:.2e}" if p < 1e-2 else f"p={p:.3f}"

# ============================
# Sortie par condition (global pour Pool)
# ============================
OUT_DIR = None

# ============================
# 1) Calcul distributions: X seul, Auto+X, Y
# ============================
def size_distrib(indiv):
    global OUT_DIR
    if OUT_DIR is None:
        raise RuntimeError("OUT_DIR is not set (condition output dir).")

    size_counter_X = np.zeros(1000, dtype=int)
    size_counter_AutoX = np.zeros(1000, dtype=int)
    size_counter_Y = np.zeros(1000, dtype=int)

    autosomes = [f"chr{i}" for i in range(1, 23)]
    chrX = "chrX"
    chrY = "chrY"

    sname = normalize_id_from_filename(indiv)

    indiv_df = pd.read_csv(
        indiv,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "score", "strand"]
    )

    try:
        indiv_df["score"] = pd.to_numeric(indiv_df["score"], errors="coerce")
        indiv_df = indiv_df.loc[indiv_df["score"].fillna(0) >= 30]
        indiv_df = indiv_df.loc[indiv_df["chr"].isin(autosomes + [chrX, chrY])]

        for _, frag in indiv_df.iterrows():
            chrom = frag["chr"]
            start = int(frag["start"])
            end = int(frag["end"])

            if chrom == chrX and is_in_par(chrom, start, end):
                continue

            size = end - start
            if not (0 <= size < 1000):
                continue

            if chrom == chrY:
                size_counter_Y[size] += 1
            elif chrom == chrX:
                size_counter_X[size] += 1
                size_counter_AutoX[size] += 1
            else:
                size_counter_AutoX[size] += 1

        total_X = size_counter_X.sum()
        total_AutoX = size_counter_AutoX.sum()
        total_Y = size_counter_Y.sum()

        norm_X = size_counter_X / total_X if total_X > 0 else np.zeros(1000)
        norm_AutoX = size_counter_AutoX / total_AutoX if total_AutoX > 0 else np.zeros(1000)
        norm_Y = size_counter_Y / total_Y if total_Y > 0 else np.zeros(1000)

        os.makedirs(OUT_DIR, exist_ok=True)
        np.savetxt(f"{OUT_DIR}/{sname}_chrX_size.tsv", norm_X, fmt="%.6f")
        np.savetxt(f"{OUT_DIR}/{sname}_chr_Auto_X_size.tsv", norm_AutoX, fmt="%.6f")
        np.savetxt(f"{OUT_DIR}/{sname}_chrY_size.tsv", norm_Y, fmt="%.6f")

        return norm_X, norm_AutoX, norm_Y

    except Exception as e:
        print(f"{sname} was skipped due to error: {e}")
        return None, None, None

# ============================
# 2) Plots
# ============================
def plot_all_indiv(files, chrom_label, output_name, cond_pretty):
    sizes = np.arange(1000)
    plt.figure(figsize=(12, 6))

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(files), 1)))
    all_arrays = []

    for idx, f in enumerate(files):
        try:
            arr = load_profile(f)
            all_arrays.append(arr)
            plt.plot(sizes, arr * 100, color=colors[idx % 20], linewidth=1, alpha=0.6)
        except Exception as e:
            print(f"Erreur lecture {f} : {e}")

    if len(all_arrays) > 0:
        mean_profile = np.mean(np.array(all_arrays), axis=0) * 100
        plt.plot(sizes, mean_profile, color="red", linewidth=2.5, label="Moyenne")

    plt.axvspan(30, 145, color="grey", alpha=0.15)
    plt.axvspan(145, 290, color="grey", alpha=0.15)
    plt.axvspan(290, 390, color="grey", alpha=0.15)

    plt.xlabel("Fragment length (bp)")
    plt.ylabel("Percent of fragments (%)")
    plt.title(f"{cond_pretty} â€” {chrom_label} (n={len(files)})")
    plt.xlim(0, 400)
    plt.ylim(0, None)

    if len(files) == 0:
        plt.close()
        print(f"Aucun fichier trouvÃ© pour {cond_pretty} {chrom_label}.")
        return

    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.close()
    print(f"Saved: {output_name}")

def plot_window(start, end, title, filename, mean_left, mean_right, label_left, label_right):
    sizes = np.arange(1000)
    plt.figure(figsize=(8, 5))

    plt.plot(sizes[start:end], mean_left[start:end] * 100, color="red", linewidth=2, label=label_left)
    plt.plot(sizes[start:end], mean_right[start:end] * 100, color="blue", linewidth=2, label=label_right)

    plt.xlabel("Fragment length (bp)")
    plt.ylabel("Percent of fragments (%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

# ============================
# 3) Boxplots
# ============================
def build_table(dir_tsv):
    x_files = sorted(glob.glob(os.path.join(dir_tsv, "*_chrX_size.tsv")))
    ax_files = sorted(glob.glob(os.path.join(dir_tsv, "*_chr_Auto_X_size.tsv")))
    y_files = sorted(glob.glob(os.path.join(dir_tsv, "*_chrY_size.tsv")))

    x_map = {sample_id_from_path(p): p for p in x_files}
    ax_map = {sample_id_from_path(p): p for p in ax_files}
    y_map = {sample_id_from_path(p): p for p in y_files}

    samples = sorted(set(x_map) & set(ax_map) & set(y_map))
    if not samples:
        raise ValueError(f"Aucun triplet X / Auto+X / Y trouvÃ© dans {dir_tsv}")

    return pd.DataFrame([{
        "sample": s,
        "path_X": x_map[s],
        "path_AutoX": ax_map[s],
        "path_Y": y_map[s]
    } for s in samples])

def compute_metrics(df_paths):
    rows = []
    for _, r in df_paths.iterrows():
        pX = load_profile(r["path_X"])
        pAX = load_profile(r["path_AutoX"])
        pY = load_profile(r["path_Y"])

        X_100_150 = frac_in_window(pX, 100, 150)
        X_150_200 = frac_in_window(pX, 150, 200)
        AX_100_150 = frac_in_window(pAX, 100, 150)
        AX_150_200 = frac_in_window(pAX, 150, 200)
        Y_100_150 = frac_in_window(pY, 100, 150)
        Y_150_200 = frac_in_window(pY, 150, 200)

        rows.append({
            "sample": r["sample"],
            "X_100_150": X_100_150,
            "X_150_200": X_150_200,
            "X_ratio": (X_100_150 / X_150_200) if X_150_200 > 0 else np.nan,
            "AutoX_100_150": AX_100_150,
            "AutoX_150_200": AX_150_200,
            "AutoX_ratio": (AX_100_150 / AX_150_200) if AX_150_200 > 0 else np.nan,
            "Y_100_150": Y_100_150,
            "Y_150_200": Y_150_200,
            "Y_ratio": (Y_100_150 / Y_150_200) if Y_150_200 > 0 else np.nan,
        })
    return pd.DataFrame(rows)

def make_boxplot(values_left, values_right, label_left, label_right,
                 ylabel, title, out_png, pvalue=None,
                 q_low=0.01, q_high=0.99, margin_frac=0.12):
    """
    Boxplot avec:
      - autoscale robuste (quantiles q_low..q_high) pour Ã©viter les boxplots "Ã©crasÃ©s"
      - p-value annotÃ©e si fournie
      - caption Wilcoxon appariÃ©
    """
    values_left = pd.Series(values_left).dropna().astype(float).values
    values_right = pd.Series(values_right).dropna().astype(float).values

    fig, ax = plt.subplots(figsize=(4.2, 5))

    bp = ax.boxplot(
        [values_left, values_right],
        labels=[label_left, label_right],
        patch_artist=True,
        widths=0.6,
        showfliers=False
    )

    colors = ["#b2239c", "#5986c1"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # --- autoscale robuste
    all_vals = np.concatenate([values_left, values_right]) if (len(values_left) + len(values_right)) else np.array([0.0])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        all_vals = np.array([0.0])

    lo = float(np.quantile(all_vals, q_low))
    hi = float(np.quantile(all_vals, q_high))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(all_vals))
        hi = float(np.nanmax(all_vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0

    yr = hi - lo
    pad = margin_frac * yr if yr > 0 else 0.1

    y_bottom = max(0.0, lo - 0.05 * yr)
    y_top_data = hi + 0.05 * yr

    # --- p-value annotation
    if pvalue is not None:
        ptxt = format_p(pvalue)
        y = y_top_data + 0.35 * pad
        ax.plot([1, 1, 2, 2], [y - 0.08 * pad, y, y, y - 0.08 * pad],
                color="black", linewidth=1.2)
        ax.text(1.5, y + 0.12 * pad, ptxt, ha="center", va="bottom", fontsize=10)
        ax.set_ylim(y_bottom, y_top_data + pad)
    else:
        ax.set_ylim(y_bottom, y_top_data)

    # --- caption
    fig.text(
        0.5, 0.01,
        "p-value calculÃ©e avec un test de Wilcoxon appariÃ© (bilatÃ©ral).",
        ha="center", va="bottom",
        fontsize=9, color="0.35"
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


# ============================
# SÃ©lection des fichiers Jiang
# ============================
def select_files(annot, cancer_label, pattern):
    filtered = annot[(annot["SEX"] == "M") & (annot["CANCER"] == cancer_label)]
    valid_ids = set(filtered["ID"].dropna().astype(str))

    all_files = sorted(glob.glob(pattern))
    kept = []
    for f in all_files:
        file_id = normalize_id_from_filename(f)
        if file_id in valid_ids:
            kept.append(f)
    return kept

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    annot = pd.read_csv("/data2/USERS/lruffinel/cfDNA/distribution_fragments/jiang/sex_cohorte_jiang.csv", sep=";")
    

    OUT_ROOT = "/data2/USERS/richaud/chrY/distribution_fragments/jiang/fragments_tsv"
    os.makedirs(OUT_ROOT, exist_ok=True)

    healthy = select_files(annot, cancer_label="Healthy", pattern="/data2/USERS/lruffinel/cfDNA/distribution_fragments/jiang/donnees_healthy/*.hg38.frag.tsv")
    cirrhosis = select_files(annot, cancer_label="Cirrhosis", pattern="/data2/USERS/richaud/finaledb/cirrhosis/jiang/*.hg38.frag.tsv")
    hepatitis = select_files(annot, cancer_label="Hepatitis B", pattern="/data2/USERS/richaud/finaledb/hepatitisb/jiang/*.hg38.frag.tsv")
    liver = select_files(annot, cancer_label="Liver", pattern="/data2/USERS/richaud/finaledb/liver/jiang/*.hg38.frag.tsv")


    # --- 4 groupes
    conditions = {
        "Healthy": {"pretty": "Healthy", "files": healthy},
        "Cirrhosis": {"pretty": "Cirrhosis", "files": cirrhosis},
        "Hepatitis B": {"pretty": "Hepatitis B", "files": hepatitis},
        "Liver": {"pretty": "Liver", "files": liver}
        }

    pool_n = 10
    batch_size = 10

    for cond_key, cfg in conditions.items():
        cond_pretty = cfg["pretty"]
        cond_out = os.path.join(OUT_ROOT, cond_key)
        os.makedirs(cond_out, exist_ok=True)

        indivs = cfg["files"]
        print(f"[{cond_pretty}] selected n={len(indivs)}")

        # 1) recalcul distributions dans le dossier de la condition
        OUT_DIR = cond_out  # global utilisÃ© par size_distrib
        for i in range(0, len(indivs), batch_size):
            batch = indivs[i:i + batch_size]
            with Pool(pool_n) as p:
                p.map(size_distrib, batch)

        # 2) courbes + fenÃªtres par condition
        files_X = sorted(glob.glob(f"{cond_out}/*chrX_size.tsv"))
        files_AutoX = sorted(glob.glob(f"{cond_out}/*chr_Auto_X_size.tsv"))
        files_Y = sorted(glob.glob(f"{cond_out}/*chrY_size.tsv"))

        plot_all_indiv(files_X, "X", os.path.join(cond_out, f"{cond_key}_size_profile_X.svg"), cond_pretty)
        plot_all_indiv(files_AutoX, "Auto+X", os.path.join(cond_out, f"{cond_key}_size_profile_Auto_X.svg"), cond_pretty)
        plot_all_indiv(files_Y, "Y", os.path.join(cond_out, f"{cond_key}_size_profile_Y.svg"), cond_pretty)

        all_X = load_matrix(files_X)
        all_AutoX = load_matrix(files_AutoX)
        all_Y = load_matrix(files_Y)

        if all_X.size == 0 or all_AutoX.size == 0 or all_Y.size == 0:
            print(f"[{cond_pretty}] WARNING: matrices vides -> skip windows/boxplots.")
            continue

        mean_X = np.mean(all_X, axis=0)
        mean_AutoX = np.mean(all_AutoX, axis=0)
        mean_Y = np.mean(all_Y, axis=0)

        plot_window(30, 145, f"{cond_pretty} â€” 30â€“145 bp fragment fraction",
                    os.path.join(cond_out, f"{cond_key}_window_30_145_X_vs_Y.svg"),
                    mean_X, mean_Y, "mean(X)", "mean(Y)")
        plot_window(145, 290, f"{cond_pretty} â€” 145â€“290 bp fragment fraction",
                    os.path.join(cond_out, f"{cond_key}_window_145_290_X_vs_Y.svg"),
                    mean_X, mean_Y, "mean(X)", "mean(Y)")
        plot_window(290, 390, f"{cond_pretty} â€” 290â€“390 bp fragment fraction",
                    os.path.join(cond_out, f"{cond_key}_window_290_390_X_vs_Y.svg"),
                    mean_X, mean_Y, "mean(X)", "mean(Y)")

        plot_window(30, 145, f"{cond_pretty} â€” 30â€“145 bp fragment fraction",
                    os.path.join(cond_out, f"{cond_key}_window_30_145_Auto_X_vs_Y.svg"),
                    mean_AutoX, mean_Y, "mean(Auto+X)", "mean(Y)")
        plot_window(145, 290, f"{cond_pretty} â€” 145â€“290 bp fragment fraction",
                    os.path.join(cond_out, f"{cond_key}_window_145_290_Auto_X_vs_Y.svg"),
                    mean_AutoX, mean_Y, "mean(Auto+X)", "mean(Y)")
        plot_window(290, 390, f"{cond_pretty} â€” 290â€“390 bp fragment fraction",
                    os.path.join(cond_out, f"{cond_key}_window_290_390_Auto_X_vs_Y.svg"),
                    mean_AutoX, mean_Y, "mean(Auto+X)", "mean(Y)")

        # 3) boxplots par condition (100â€“150, 150â€“200, ratio)
        df_paths = build_table(cond_out)
        df = compute_metrics(df_paths)

        make_boxplot(df["Y_100_150"], df["X_100_150"], "Y", "X",
                     "Proportion of fragments", f"{cond_pretty} â€” 100â€“150 bp",
                     os.path.join(cond_out, f"{cond_key}_boxplot_100_150_Y_vs_X.svg"))
        make_boxplot(df["Y_150_200"], df["X_150_200"], "Y", "X",
                     "Proportion of fragments", f"{cond_pretty} â€” 150â€“200 bp",
                     os.path.join(cond_out, f"{cond_key}_boxplot_150_200_Y_vs_X.svg"))
        make_boxplot(df["Y_ratio"], df["X_ratio"], "Y", "X",
                     "Ratio", f"{cond_pretty} â€” Short/Long Ratio (100â€“150 / 150â€“200)",
                     os.path.join(cond_out, f"{cond_key}_boxplot_ratio_Y_vs_X.svg"))

        make_boxplot(df["Y_100_150"], df["AutoX_100_150"], "Y", "Auto+X",
                     "Proportion of fragments", f"{cond_pretty} â€” 100â€“150 bp",
                     os.path.join(cond_out, f"{cond_key}_boxplot_100_150_Y_vs_Auto_X.svg"))
        make_boxplot(df["Y_150_200"], df["AutoX_150_200"], "Y", "Auto+X",
                     "Proportion of fragments", f"{cond_pretty} â€” 150â€“200 bp",
                     os.path.join(cond_out, f"{cond_key}_boxplot_150_200_Y_vs_Auto_X.svg"))
        make_boxplot(df["Y_ratio"], df["AutoX_ratio"], "Y", "Auto+X",
                     "Ratio", f"{cond_pretty} â€” Short/Long Ratio (100â€“150 / 150â€“200)",
                     os.path.join(cond_out, f"{cond_key}_boxplot_ratio_Y_vs_Auto_X.svg"))

    print("TerminÃ©: un dossier par condition avec toutes les figures.")
