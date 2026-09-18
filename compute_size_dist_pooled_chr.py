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

def normalize_id_from_filename(path: str) -> str:
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

# ============================
# 1) Extraction Worker (Saves Raw Integer Counts)
# ============================
def process_patient_file_raw_counts(indiv_path: str, out_dir: str, target_chromosomes: list):
    sname = normalize_id_from_filename(indiv_path)
    out_tsv = os.path.join(out_dir, f"{sname}_raw_counts.tsv")

    if os.path.exists(out_tsv):
        return True

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
        sizes_pooled = sizes[valid_size_mask].to_numpy()

        os.makedirs(out_dir, exist_ok=True)
        counts = np.bincount(sizes_pooled, minlength=1000).astype(np.int64)

        # Save integer counts directly
        np.savetxt(out_tsv, counts, fmt="%d")
        return True

    except Exception as e:
        print(f"Error processing {sname}: {e}")
        return False

# ============================
# 2) File Selector (Male filter)
# ============================
def select_male_files(annot: pd.DataFrame, cancer_label: str, pattern: str) -> list:
    annot["Sexe"] = annot["Sexe"].astype(str).str.strip()
    annot["Cancer"] = annot["Cancer"].astype(str).str.strip()
    filtered = annot[(annot["Sexe"] == "M") & (annot["Cancer"] == cancer_label)]
    valid_ids = set(filtered["ID_FinaleDB"].dropna().astype(str).str.strip())

    all_files = sorted(glob.glob(pattern))
    return [f for f in all_files if normalize_id_from_filename(f) in valid_ids]

# ============================
# 3) Plotting
# ============================
def plot_conditions_overlay(condition_data: dict, out_filename: str, xlim=(30, 250)):
    sizes = np.arange(1000)
    plt.figure(figsize=(10, 6))

    colors = {
        "Healthy": "#2ca02c",
        "Lung Cancer": "#1f77b4",
        "Pancreatic Cancer": "#ff7f0e",
        "Colorectal Cancer": "#d62728",
    }

    start, end = xlim
    for cond_name, metrics in condition_data.items():
        density = metrics["density"]
        total_frags = metrics["total_fragments"]
        n_patients = metrics["n_patients"]

        color = colors.get(cond_name, None)
        label_text = f"{cond_name} (n={n_patients}, {total_frags:,} frags)"
        
        plt.plot(
            sizes[start:end],
            density[start:end] * 100,
            label=label_text,
            linewidth=2.2,
            color=color,
        )

    plt.xlabel("Fragment length (bp)", fontsize=12)
    plt.ylabel("Percent of fragments (%)", fontsize=12)
    plt.title("Male Fragment Size Distribution — All Chromosomes & Fragments Pooled", fontsize=13, fontweight="bold")
    plt.xlim(start, end)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_filename, format="svg", dpi=300)
    plt.close()
    print(f"Saved: {out_filename}")

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    annot = pd.read_csv(
        "/data2/USERS/lruffinel/cfDNA/distribution_fragments/cristiano/cristiano_patients_annotation.csv",
        sep=";",
    )

    OUT_ROOT = "/data2/USERS/richaud/distribution_fragments/cristiano_males_grand_pool/"
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
    autosomes = [f"chr{i}" for i in range(1, 23)]
    chromosomes = autosomes + ["chrX", "chrY"]

    condition_results = {}

    for cond_key, cfg in conditions.items():
        cond_pretty = cfg["pretty"]
        cond_out = os.path.join(OUT_ROOT, cond_key)
        os.makedirs(cond_out, exist_ok=True)

        indivs = select_male_files(annot, cfg["cancer_label"], cfg["pattern"])
        print(f"\n================ [{cond_pretty}] Selected Male Samples: n={len(indivs)} ================")

        worker = partial(
            process_patient_file_raw_counts,
            out_dir=cond_out,
            target_chromosomes=chromosomes,
        )

        for i in range(0, len(indivs), batch_size):
            batch = indivs[i:i + batch_size]
            with Pool(pool_n) as p:
                p.map(worker, batch)

        # Step 2: Sum raw fragment counts across all patients in condition
        tsv_files = sorted(glob.glob(os.path.join(cond_out, "*_raw_counts.tsv")))
        
        pooled_counts = np.zeros(1000, dtype=np.int64)
        usable_patients = 0

        for f in tsv_files:
            arr = np.loadtxt(f, dtype=np.int64)
            if arr.sum() > 0:
                pooled_counts += arr
                usable_patients += 1

        grand_total = pooled_counts.sum()
        print(f"[{cond_pretty}] Included {usable_patients} patients | Total pooled fragments: {grand_total:,}")

        if grand_total > 0:
            # Divide condition-wide sum by the condition-wide grand total
            global_density = pooled_counts / grand_total
            condition_results[cond_pretty] = {
                "density": global_density,
                "total_fragments": grand_total,
                "n_patients": usable_patients,
            }

            # Optional: Save pooled counts and pooled density for this condition
            np.savetxt(os.path.join(cond_out, f"{cond_key}_grand_pool_counts.tsv"), pooled_counts, fmt="%d")
            np.savetxt(os.path.join(cond_out, f"{cond_key}_grand_pool_density.tsv"), global_density, fmt="%.8f")

    # Step 3: Combined plot
    if condition_results:
        plot_out = os.path.join(OUT_ROOT, "males_grand_pool_condition_comparison.svg")
        plot_conditions_overlay(condition_results, plot_out, xlim=(30, 250))