import pandas as pd
import numpy as np
import glob
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt

# ============================
# Target Features & PAR Regions (chrX)
# ============================
ALL_FEATURES = ["Ampliconic", "X-transposed", "X-degenerate", "Heterochromatic", "Pseudo-autosomal", "Others"]


PAR_X = [
    (10_001 - 1, 2_781_479),
    (155_701_383 - 1, 156_030_895),
]

def is_in_par_x(start: int, end: int) -> bool:
    for p_start, p_end in PAR_X:
        if start < p_end and p_start < end:
            return True
    return False

# ============================
# Fast Interval Indexer per Chromosome
# ============================
def load_interval_tree(annot_filepath: str):
    """
    Parses BED annotation file into structured NumPy arrays using integer feature codes.
    """
    df = pd.read_csv(
        annot_filepath,
        sep=r"\s+",
        header =0,
        engine="c"
    )

    df["chromosome"] = df["chromosome"].astype(str).apply(lambda c: c if c.startswith("chr") else f"chr{c}")
    
    # Map feature labels into standardized categories
    feat_lower_map = {f.lower(): f for f in ALL_FEATURES}

    def assign_feature(annot_str):
        s = str(annot_str).lower()
        for key, feat in feat_lower_map.items():
            if key in s:
                return feat
        return "Unknown"

    df["feature"] = df["Subregion_type"].apply(assign_feature)

    # Convert features to numerical codes for vectorized lookups (-1 for unmapped)
    feat_to_id = {f: i for i, f in enumerate(ALL_FEATURES)}
    df["feat_id"] = df["feature"].map(feat_to_id).fillna(-1).astype(np.int16)

    chrom_tables = {}
    for chrom, group in df.groupby("chromosome"):
        group_sorted = group.sort_values("start")
        chrom_tables[chrom] = {
            "starts": group_sorted["start"].to_numpy(dtype=np.int64),
            "ends": group_sorted["end"].to_numpy(dtype=np.int64),
            "feat_id": group_sorted["feat_id"].to_numpy(dtype=np.int16),
        }

    return chrom_tables

# Global cache for worker processes
CHROM_TABLES = None
ANNOT_PATH = "chry_annot_2023.txt"
OUT_DIR = None

def normalize_id_from_filename(path):
    base = os.path.basename(path)
    for suf in [".hg38.frag.tsv", ".hg38.frag.bed", ".hg38.frag", ".frag.bed", ".frag", ".bed", ".tsv"]:
        if base.endswith(suf):
            return base[: -len(suf)]
    return os.path.splitext(base)[0]

# ============================
# Vectorized Profile Extractor
# ============================
def process_single_patient(indiv):
    global OUT_DIR, CHROM_TABLES, ANNOT_PATH
    if OUT_DIR is None:
        raise RuntimeError("OUT_DIR is not set.")

    if CHROM_TABLES is None:
        CHROM_TABLES = load_interval_tree(ANNOT_PATH)

    all_chroms = ["chr17", "chr19", "chr22", "chrY"]
    sname = normalize_id_from_filename(indiv)

    # Pre-allocate fragment length bins: [feature_count, length_bins]
    num_feats = len(ALL_FEATURES)
    counters = {
        c: np.zeros((num_feats, 1000), dtype=np.int64) 
        for c in all_chroms
    }
    totals = {
        c: np.zeros(1000, dtype=np.int64) 
        for c in all_chroms
    }

    try:
        indiv_df = pd.read_csv(
            indiv,
            sep="\t",
            header=None,
            usecols=[0, 1, 2, 3],
            names=["chr", "start", "end", "score"],
            engine="c"
        )
    except Exception as e:
        print(f"Error reading {indiv}: {e}")
        return False
    
    indiv_df["score"] = pd.to_numeric(indiv_df["score"], errors="coerce")
    indiv_df = indiv_df.loc[indiv_df["score"].fillna(0) >= 30].copy()
    
    indiv_df["chr"] = indiv_df["chr"].astype(str).apply(lambda c: c if c.startswith("chr") else f"chr{c}")
    indiv_df = indiv_df.loc[indiv_df["chr"].isin(all_chroms)].copy()

    for chrom, frags in indiv_df.groupby("chr"):
        if chrom not in CHROM_TABLES:
            continue

        ref_starts = CHROM_TABLES[chrom]["starts"]
        ref_ends = CHROM_TABLES[chrom]["ends"]
        ref_feat_id = CHROM_TABLES[chrom]["feat_id"]

        frag_starts = frags["start"].to_numpy(dtype=np.int64)
        frag_ends = frags["end"].to_numpy(dtype=np.int64)
        frag_mid = (frag_starts + frag_ends) // 2

        # Binary search containing reference interval
        idx = np.searchsorted(ref_starts, frag_mid, side="right") - 1

        valid_mask = (idx >= 0) & (idx < len(ref_starts))
        idx_clipped = np.clip(idx, 0, len(ref_starts) - 1)
        overlap_mask = valid_mask & (frag_starts < ref_ends[idx_clipped]) & (ref_starts[idx_clipped] < frag_ends)

        sizes = frag_ends - frag_starts
        valid_sizes = (sizes >= 30) & (sizes < 250)

        if chrom == "chrX":
            par_mask = np.vectorize(is_in_par_x)(frag_starts, frag_ends)
            keep_mask = valid_sizes & (~par_mask)
        else:
            keep_mask = valid_sizes

        # Assign feature IDs
        frag_features = np.full(len(frag_starts), -1, dtype=np.int16)
        frag_features[overlap_mask] = ref_feat_id[idx_clipped[overlap_mask]]

        # Retain filtered fragments
        final_sizes = sizes[keep_mask]
        final_feats = frag_features[keep_mask]

        # Aggregate per-feature profiles
        for f_idx in range(num_feats):
            matched = (final_feats == f_idx)
            if np.any(matched):
                sz, cnt = np.unique(final_sizes[matched], return_counts=True)
                counters[chrom][f_idx, sz] += cnt

        # Track total valid fragments on this chromosome for 'Other' calculation
        all_sz, all_cnt = np.unique(final_sizes, return_counts=True)
        totals[chrom][all_sz] += all_cnt

    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Save normalized distributions (Feature vs. Other)
    for chrom in all_chroms:
        chrom_total_counts = totals[chrom]
        for f_idx, feat in enumerate(ALL_FEATURES):
            feat_counts = counters[chrom][f_idx]
            other_counts = chrom_total_counts - feat_counts

            # Normalized Feature Array
            tot_feat = feat_counts.sum()
            norm_feat = feat_counts / tot_feat if tot_feat > 0 else np.zeros(1000)
            np.savetxt(f"{OUT_DIR}/{sname}_{chrom}_{feat}_size_hallast.tsv", norm_feat, fmt="%.6f")

            # Normalized Complement Array (Other)
            tot_other = other_counts.sum()
            norm_other = other_counts / tot_other if tot_other > 0 else np.zeros(1000)
            np.savetxt(f"{OUT_DIR}/{sname}_{chrom}_{feat}_Other_size_hallast.tsv", norm_other, fmt="%.6f")

    return True

# ============================
# Helpers & Plotting
# ============================
def load_profile(tsv_path):
    arr = np.loadtxt(tsv_path)
    if arr.shape[0] != 1000:
        raise ValueError(f"Unexpected shape: {tsv_path} -> {arr.shape}")
    return arr

def load_matrix(files):
    mats = []
    for f in files:
        try:
            mats.append(load_profile(f))
        except Exception as e:
            print(f"Skip {f} : {e}")
    return np.array(mats)

def plot_chrom_multi_features(
    cond_out,
    chrom_label,
    cond_pretty,
    cond_key,
    target_feats,
    all_feats,
    output_plot_name,
    output_tsv_dir,
):
  """Plots target features + combined Other, and exports individual length profile TSVs."""
  sizes = np.arange(30, 250)
  plt.figure(figsize=(10, 5.5))
  os.makedirs(output_tsv_dir, exist_ok=True)

  palette = {
      "Ampliconic": "#D95F02",
      "X-transposed": "#7570B3",
      "X-degenerate": "#E7298A",
      "Heterochromatic": "#16711935",
      "Pseudo-autosomal": "#16776960",
      "Other": "#666666",
  }

  # 1. Process and save each target feature (HET, PromF, TSS)
  for feat in target_feats:
    files = sorted(glob.glob(f"{cond_out}/*_{chrom_label}_{feat}_size_hallast.tsv"))
    if not files:
      continue

    matrix = load_matrix(files)
    if matrix.size > 0:
      mean_profile = np.mean(matrix, axis=0)  # Shape (1000,)

      # Save full length distribution (bins 0 to 999)
      out_tsv = os.path.join(
          output_tsv_dir, f"{cond_key}_{chrom_label}_{feat}_mean_size_hallast.tsv"
      )
      np.savetxt(out_tsv, mean_profile, fmt="%.6f")

      # Plot selected window [30:250] in percentage
      plt.plot(
          sizes,
          mean_profile[30:250] * 100,
          color=palette.get(feat, "#333333"),
          linewidth=2.2,
          label=feat,
      )

  # 2. Process and save combined Other (all non-target features)
  other_feats = [f for f in all_feats if f not in target_feats]
  other_matrices = []

  for feat in other_feats:
    files = sorted(glob.glob(f"{cond_out}/*_{chrom_label}_{feat}_size_hallast.tsv"))
    if files:
      mat = load_matrix(files)
      if mat.size > 0:
        other_matrices.append(mat)

  if other_matrices:
    combined_other = np.mean(np.concatenate(other_matrices, axis=0), axis=0)

    # Save Other distribution
    other_tsv = os.path.join(
        output_tsv_dir, f"{cond_key}_{chrom_label}_Other_mean_size_hallast.tsv"
    )
    np.savetxt(other_tsv, combined_other, fmt="%.6f")

    # Plot Other curve
    plt.plot(
        sizes,
        combined_other[30:250] * 100,
        color=palette["Other"],
        linewidth=2.0,
        linestyle="--",
        label="Other",
    )

  # Visual styling
  plt.axvspan(30, 145, color="grey", alpha=0.12)
  plt.axvspan(145, 250, color="grey", alpha=0.08)

  plt.xlabel("Fragment length (bp)", fontsize=11)
  plt.ylabel("Percent of fragments (%)", fontsize=11)
  plt.title(
      f"{cond_pretty} — {chrom_label}: Chromatin Features vs Other",
      fontsize=13,
      fontweight="bold",
  )
  plt.xlim(30, 250)
  plt.ylim(0, None)
  plt.legend(title="Feature", loc="upper right", frameon=True)
  plt.tight_layout()

  plt.savefig(output_plot_name, format="svg", dpi=300)
  plt.close()

# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    ANNOT_PATH = "chry_annot_2023.txt"
    print("Pre-loading annotation lookup tables...")
    CHROM_TABLES = load_interval_tree(ANNOT_PATH)

    annot = pd.read_csv(
        "/data2/USERS/lruffinel/cfDNA/distribution_fragments/cristiano/cristiano_patients_annotation.csv",
        sep=";"
    )

    OUT_ROOT = "/data2/USERS/richaud/distribution_fragments_het/cristiano/"
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
    batch_size = 10

    all_chroms = [f"chr{i}" for i in ["Y"]]

    for cond_key, cfg in conditions.items():
        cond_pretty = cfg["pretty"]
        cond_out = os.path.join(OUT_ROOT, cond_key)
        os.makedirs(cond_out, exist_ok=True)

        filtered = annot[(annot["Sexe"] == "M") & (annot["Cancer"] == cfg["cancer_label"])]
        valid_ids = set(filtered["ID_FinaleDB"].dropna().astype(str))
        all_files = sorted(glob.glob(cfg["pattern"]))
        indivs = [f for f in all_files if normalize_id_from_filename(f) in valid_ids]
        
        print(f"\n[{cond_pretty}] selected n={len(indivs)}")

        OUT_DIR = cond_out
        for i in range(0, len(indivs), batch_size):
            batch = indivs[i:i + batch_size]
            with Pool(pool_n) as p:
                p.map(process_single_patient, batch)
            print(f"[{cond_pretty}] Processed batch starting at index {i}")

        # Generating Output Plots
        # Generating Output Plots & Summary TSVs
        plots_chrom_dir = os.path.join(cond_out, "per_chromosome_summary")
        tsvs_chrom_dir = os.path.join(cond_out, "summary_profiles_tsv")
        os.makedirs(plots_chrom_dir, exist_ok=True)
        os.makedirs(tsvs_chrom_dir, exist_ok=True)

        target_features = ["Ampliconic", "X-transposed", "X-degenerate", "Heterochromatic", "Pseudo-autosomal"]

        for chrom in all_chroms:
          plot_chrom_multi_features(
              cond_out=cond_out,
              chrom_label=chrom,
              cond_pretty=cond_pretty,
              cond_key=cond_key,
              target_feats=target_features,
              all_feats=ALL_FEATURES,
              output_plot_name=os.path.join(
                  plots_chrom_dir,
                  f"{cond_key}_{chrom}_hallast.svg",
              ),
              output_tsv_dir=tsvs_chrom_dir,
          )

        print(f"[{cond_pretty}] Summary plots and profile TSVs generated.")