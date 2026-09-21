import glob
import os
from functools import partial
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================
# PAR Coordinates (hg38 0-based half-open)
# ============================
PAR = {
    "X": [
        (10_000, 2_781_479),          # PAR1
        (155_701_382, 156_030_895),  # PAR2
    ],
    "Y": [
        (10_000, 2_781_479),          # PAR1
        (56_887_901, 57_217_415),    # PAR2
    ],
}

def norm_chrom(chrom_col: pd.Series) -> pd.Series:
    return chrom_col.astype(str).str.strip().str.replace(r"^chr", "", regex=True)

def filter_par(df: pd.DataFrame) -> pd.DataFrame:
    norm_c = norm_chrom(df["chr"])
    is_par = pd.Series(False, index=df.index)

    for chrom, intervals in PAR.items():
        chr_mask = norm_c == chrom
        for p_start, p_end in intervals:
            in_window = (df["start"] < p_end) & (df["end"] > p_start)
            is_par = is_par | (chr_mask & in_window)

    return df.loc[~is_par]

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

def load_profile(tsv_path):
    arr = np.loadtxt(tsv_path)
    if arr.shape[0] != 1000:
        raise ValueError(f"Unexpected array size in {tsv_path}: {arr.shape}")
    return arr

# ============================
# 1) Extraction Worker (Single-Pass)
# ============================
def process_patient_file(indiv_path: str, out_dir: str, target_chromosomes: list, min_reads_threshold: int = 100):
    sname = normalize_id_from_filename(indiv_path)
    try:
        df = pd.read_csv(
            indiv_path,
            sep="\t",
            header=None,
            names=["chr", "start", "end", "score"],
            usecols=[0, 1, 2, 3],
            dtype={"chr": str, "start": "int64", "end": "int64", "score": str},
        )

        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
        df = df.loc[df["score"] >= 30]

        df["chr_clean"] = norm_chrom(df["chr"])
        valid_clean = [norm_chrom(pd.Series([c])).iloc[0] for c in target_chromosomes]
        df = df.loc[df["chr_clean"].isin(valid_clean)]
        df = filter_par(df)

        sizes = df["end"] - df["start"]
        valid_size_mask = (sizes >= 0) & (sizes < 1000)
        df = df.loc[valid_size_mask]
        df["size"] = sizes[valid_size_mask]

        os.makedirs(out_dir, exist_ok=True)
        meta_counts = []

        for chrom_raw in target_chromosomes:
            chrom_clean = norm_chrom(pd.Series([chrom_raw])).iloc[0]

            mask_focal = df["chr_clean"] == chrom_clean
            sizes_focal = df.loc[mask_focal, "size"].to_numpy()
            sizes_others = df.loc[~mask_focal, "size"].to_numpy()

            counts_focal = np.bincount(sizes_focal, minlength=1000)
            counts_others = np.bincount(sizes_others, minlength=1000)

            total_focal = counts_focal.sum()
            total_others = counts_others.sum()

            norm_focal = counts_focal / total_focal if total_focal > 0 else np.zeros(1000)
            norm_others = counts_others / total_others if total_others > 0 else np.zeros(1000)

            np.savetxt(os.path.join(out_dir, f"{sname}_chr{chrom_raw}_female_size.tsv"), norm_focal, fmt="%.6f")
            np.savetxt(os.path.join(out_dir, f"{sname}_chr{chrom_raw}_others_female_size.tsv"), norm_others, fmt="%.6f")

            meta_counts.append({
                "sample": sname,
                "chrom": chrom_raw,
                "raw_count_focal": total_focal,
                "raw_count_others": total_others,
                "is_usable": total_focal >= min_reads_threshold
            })

        pd.DataFrame(meta_counts).to_csv(
            os.path.join(out_dir, f"{sname}_counts_meta.csv"), index=False
        )
        return True

    except Exception as e:
        print(f"Error processing {sname}: {e}")
        return False

# ============================
# 2) Plotting Helpers
# ============================
def plot_window_comparison(start, end, title, filename, mean_focal, mean_others, label_focal, label_others):
    sizes = np.arange(1000)
    plt.figure(figsize=(9, 5))

    plt.plot(sizes[start:end], mean_others[start:end] * 100, color="red", linewidth=2, label=label_others)
    plt.plot(sizes[start:end], mean_focal[start:end] * 100, color="blue", linewidth=2, label=label_focal)

    plt.xlabel("Fragment length (bp)", fontsize=11)
    plt.ylabel("Percent of fragments (%)", fontsize=11)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(filename, format="svg", dpi=300)
    plt.close()
    print(f"Saved: {filename}")

def plot_all_individuals(files, chrom_label, output_name, cond_pretty):
    sizes = np.arange(1000)
    plt.figure(figsize=(10, 6))

    all_arrays = []
    for f in files:
        try:
            arr = load_profile(f)
            if arr.sum() > 0:  # Exclude flat zero files
                all_arrays.append(arr)
                plt.plot(sizes, arr * 100, color="gray", linewidth=0.7, alpha=0.35)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if len(all_arrays) > 0:
        mean_profile = np.mean(np.array(all_arrays), axis=0) * 100
        plt.plot(sizes, mean_profile, color="crimson", linewidth=2.5, label=f"Mean (n={len(all_arrays)})")

    plt.axvspan(30, 145, color="grey", alpha=0.15)
    plt.axvspan(145, 290, color="grey", alpha=0.15)
    plt.xlabel("Fragment length (bp)", fontsize=11)
    plt.ylabel("Percent of fragments (%)", fontsize=11)
    plt.title(f"{cond_pretty} — {chrom_label}", fontsize=12, fontweight="bold")
    plt.xlim(0, 400)
    plt.ylim(0, None)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_name, format="svg", dpi=300)
    plt.close()
    print(f"Saved: {output_name}")

# ============================
# 3) File Selector
# ============================
def select_files(annot, cancer_label, pattern):
    annot["Sexe"] = annot["Sexe"].astype(str).str.strip()
    annot["Cancer"] = annot["Cancer"].astype(str).str.strip()
    filtered = annot[(annot["Sexe"] == "M") & (annot["Cancer"] == cancer_label)]
    valid_ids = set(filtered["ID_FinaleDB"].dropna().astype(str).str.strip())

    all_files = sorted(glob.glob(pattern))
    return [f for f in all_files if normalize_id_from_filename(f) in valid_ids]


# ============================
# 3) Boxplots
# ============================

def sample_id_from_path(path):
    return os.path.basename(path).split("_chr")[0]

def frac_in_window(profile, start_bp, end_bp):
    return float(np.sum(profile[start_bp:end_bp]))

def build_table(dir_tsv, chrom):
    ax_files = sorted(glob.glob(os.path.join(dir_tsv, f"*_chr{chrom}_others_size.tsv")))
    y_files = sorted(glob.glob(os.path.join(dir_tsv, f"*_chr{chrom}_size.tsv")))
    print(ax_files)

    ax_map = {sample_id_from_path(p): p for p in ax_files}
    y_map = {sample_id_from_path(p): p for p in y_files}
    print(ax_map)

    samples = sorted(set(ax_map) & set(y_map))
    if not samples:
        raise ValueError(f"Aucun triplet X / Auto+X / Y trouvÃ© dans {dir_tsv}")

    return pd.DataFrame([{
        "sample": s,
        "path_AutoX": ax_map[s],
        "path_Y": y_map[s]
    } for s in samples])

def compute_metrics(df_paths):
    rows = []
    for _, r in df_paths.iterrows():
        pAX = load_profile(r["path_AutoX"])
        pY = load_profile(r["path_Y"])

        AX_100_150 = frac_in_window(pAX, 100, 165)
        AX_150_200 = frac_in_window(pAX, 165, 210)
        Y_100_150 = frac_in_window(pY, 100, 165)
        Y_150_200 = frac_in_window(pY, 165, 210)

        rows.append({
            "sample": r["sample"],
            "AutoX_100_150": AX_100_150,
            "AutoX_150_200": AX_150_200,
            "AutoX_ratio": (AX_100_150 / AX_150_200) if AX_150_200 > 0 else np.nan,
            "Y_100_150": Y_100_150,
            "Y_150_200": Y_150_200,
            "Y_ratio": (Y_100_150 / Y_150_200) if Y_150_200 > 0 else np.nan,
        })
    return pd.DataFrame(rows)

def format_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p=NA"
    if p < 1e-4:
        return "p<1e-4"
    return f"p={p:.2e}" if p < 1e-2 else f"p={p:.3f}"

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
# MAIN
# ============================
if __name__ == "__main__":
    annot = pd.read_csv(
        "/data2/USERS/lruffinel/cfDNA/distribution_fragments/cristiano/cristiano_patients_annotation.csv",
        sep=";",
    )

    OUT_ROOT = "/data2/USERS/richaud/chrY/distribution_fragments/cristiano/"
    os.makedirs(OUT_ROOT, exist_ok=True)

    conditions = {
        "healthy": {
            "pretty": "Healthy",
            "pattern": "/data2/USERS/BIOINFO/databases/FinaleDB/cristiano-data-2019/*.tsv",
            "cancer_label": "Healthy",
        },
        "lung_cancer": {
            "pretty": "Lung Cancer",
            "pattern": "/data2/USERS/richaud/finaledb/lung/cristiano/indivs/*.hg38.frag",
            "cancer_label": "Lung Cancer",
        },
        "pancreatic_cancer": {
            "pretty": "Pancreatic Cancer",
            "pattern": "/data2/USERS/richaud/finaledb/pancreatic/cristiano/indivs/*.hg38.frag.bed",
            "cancer_label": "Pancreatic Cancer",
        },
        "colorectal_cancer": {
            "pretty": "Colorectal Cancer",
            "pattern": "/data2/USERS/richaud/finaledb/colorectal/cristiano/indivs/*.hg38.frag.bed",
            "cancer_label": "Colorectal Cancer",
        },
    }

    pool_n = 10
    batch_size = 20
    MIN_READS_THRESHOLD = 100  # Excludes near-empty chrY profiles from pulling down the cohort mean

    autosomes = [f"{i}" for i in ["Y"]]
    chromosomes = autosomes 

    for cond_key, cfg in conditions.items():
        cond_pretty = cfg["pretty"]
        cond_out = os.path.join(OUT_ROOT, cond_key)
        os.makedirs(cond_out, exist_ok=True)

        indivs = select_files(annot, cfg["cancer_label"], cfg["pattern"])
        print(f"\n================ [{cond_pretty}] Selected Male Samples: n={len(indivs)} ================")

        # Step 1: Single-pass file extraction
        worker = partial(
            process_patient_file,
            out_dir=cond_out,
            target_chromosomes=chromosomes,
            min_reads_threshold=MIN_READS_THRESHOLD,
        )

        for i in range(0, len(indivs), batch_size):
            batch = indivs[i:i + batch_size]
            with Pool(pool_n) as p:
                p.map(worker, batch)
            print(f"[{cond_pretty}] Processed batch {i} to {min(i + batch_size, len(indivs))}")

        # Step 2: Plotting for each chromosome
        for chrom in chromosomes:
            files_focal = sorted(glob.glob(os.path.join(cond_out, f"*_chr{chrom}_size.tsv")))
            files_others = sorted(glob.glob(os.path.join(cond_out, f"*_chr{chrom}_others_size.tsv")))

            # Load only valid, non-zero arrays
            arrays_focal = [load_profile(f) for f in files_focal if load_profile(f).sum() > 0]
            arrays_others = [load_profile(f) for f in files_others if load_profile(f).sum() > 0]

            if len(arrays_focal) == 0 or len(arrays_others) == 0:
                print(f"[{cond_pretty}] Skipping {chrom}: No valid profiles found above threshold.")
                continue

            mean_focal = np.mean(np.array(arrays_focal), axis=0)
            mean_others = np.mean(np.array(arrays_others), axis=0)

            # Window Comparison Plot (30 to 250 bp)
            plot_window_comparison(
                start=30,
                end=250,
                title=f"{cond_pretty} — {chrom} vs Others (30–250 bp)",
                filename=os.path.join(cond_out, f"{cond_key}_window_30_250_female_{chrom}.svg"),
                mean_focal=mean_focal,
                mean_others=mean_others,
                label_focal=f"mean({chrom}) [n={len(arrays_focal)}]",
                label_others=f"mean({chrom} others) [n={len(arrays_others)}]",
            )

            # cond_out = "/data2/USERS/richaud/chrY/distribution_fragments/cristiano/healthy"
            # df_paths = build_table(cond_out, chrom)
            # df = compute_metrics(df_paths)

            # make_boxplot(df["Y_ratio"], df["AutoX_ratio"], f"{chrom}", "Others",
            #             "Ratio", f"{cond_pretty} â€” Short/Long Ratio (100â€“150 / 150â€“200)",
            #             os.path.join(cond_out, f"{cond_key}_boxplot_ratio_{chrom}_vs_others.svg"))
            
            # df_paths = build_table("/data2/USERS/richaud/chrY/distribution_fragments/cristiano/colorectal_cancer", chrom)
            # df = compute_metrics(df_paths)

            # make_boxplot(df["Y_ratio"], df["AutoX_ratio"], f"{chrom}", "Others",
            #             "Ratio", f"{cond_pretty} â€” Short/Long Ratio (100â€“150 / 150â€“200)",
            #             os.path.join(cond_out, f"{cond_key}_boxplot_ratio_{chrom}_vs_others.svg"))
            

            # df_paths = build_table("/data2/USERS/richaud/chrY/distribution_fragments/cristiano/pancreatic_cancer", chrom)
            # df = compute_metrics(df_paths)

            # make_boxplot(df["Y_ratio"], df["AutoX_ratio"], f"{chrom}", "Others",
            #             "Ratio", f"{cond_pretty} â€” Short/Long Ratio (100â€“150 / 150â€“200)",
            #             os.path.join(cond_out, f"{cond_key}_boxplot_ratio_{chrom}_vs_others.svg"))
            
            # df_paths = build_table("/data2/USERS/richaud/chrY/distribution_fragments/cristiano/lung_cancer", chrom)
            # df = compute_metrics(df_paths)

            # make_boxplot(df["Y_ratio"], df["AutoX_ratio"], f"{chrom}", "Others",
            #             "Ratio", f"{cond_pretty} â€” Short/Long Ratio (100â€“150 / 150â€“200)",
            #             os.path.join(cond_out, f"{cond_key}_boxplot_ratio_{chrom}_vs_others.svg"))