import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon


# ============================
# Helpers
# ============================
def load_profile(tsv_path: str) -> np.ndarray:
    """Charge un profil (1000 valeurs) depuis un *_size.tsv."""
    arr = np.loadtxt(tsv_path)
    if arr.shape[0] != 1000:
        raise ValueError(f"Taille inattendue: {tsv_path} -> {arr.shape}")
    return arr


def frac_in_window(profile: np.ndarray, start_bp: int, end_bp: int) -> float:
    """Somme des fractions entre [start_bp, end_bp[."""
    return float(np.sum(profile[start_bp:end_bp]))


def sample_id_from_path(path: str) -> str:
    """Extrait l'ID échantillon à partir de <sample>_chrX_size.tsv etc."""
    return os.path.basename(path).split("_chr")[0]


def format_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p=NA"
    if p < 1e-4:
        return "p<1e-4"
    return f"p={p:.2e}" if p < 1e-2 else f"p={p:.3f}"


def wilcoxon_paired(a, b) -> float:
    """
    Wilcoxon signed-rank test apparié (bilatéral).
    Retourne la p-value (float) ou np.nan si non calculable.
    """
    a = pd.Series(a).astype(float)
    b = pd.Series(b).astype(float)

    mask = ~(a.isna() | b.isna())
    a = a[mask].values
    b = b[mask].values

    if len(a) < 3:
        return np.nan

    # si toutes les différences sont nulles, p=1
    if np.allclose(a - b, 0):
        return 1.0

    try:
        stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(p)
    except Exception:
        return np.nan


# ============================
# Build table of triplets in a directory
# ============================
def build_table(dir_tsv: str) -> pd.DataFrame:
    """
    Cherche dans dir_tsv les fichiers:
      *_chrX_size.tsv
      *_chr_Auto_X_size.tsv
      *_chrY_size.tsv
    et retourne un tableau (sample, path_X, path_AutoX, path_Y)
    pour les samples ayant les 3.
    """
    x_files = sorted(glob.glob(os.path.join(dir_tsv, "*_chrX_size.tsv")))
    ax_files = sorted(glob.glob(os.path.join(dir_tsv, "*_chr_Auto_X_size.tsv")))
    y_files = sorted(glob.glob(os.path.join(dir_tsv, "*_chrY_size.tsv")))

    x_map = {sample_id_from_path(p): p for p in x_files}
    ax_map = {sample_id_from_path(p): p for p in ax_files}
    y_map = {sample_id_from_path(p): p for p in y_files}

    samples = sorted(set(x_map) & set(ax_map) & set(y_map))
    if not samples:
        raise ValueError(f"Aucun triplet X / Auto+X / Y trouvé dans {dir_tsv}")

    return pd.DataFrame([{
        "sample": s,
        "path_X": x_map[s],
        "path_AutoX": ax_map[s],
        "path_Y": y_map[s],
    } for s in samples])


# ============================
# Compute metrics for boxplots
# ============================
def compute_metrics(df_paths: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule (par sample):
      - proportions 100–147, 150–200
      - ratio (100–150)/(150–200)
    pour X, Auto+X, Y.
    """
    rows = []
    for _, r in df_paths.iterrows():
        pX = load_profile(r["path_X"])
        pAX = load_profile(r["path_AutoX"])
        pY = load_profile(r["path_Y"])

        X_100_147 = frac_in_window(pX, 100, 147)
        X_147_200 = frac_in_window(pX, 147, 200)
        AX_100_147 = frac_in_window(pAX, 100, 147)
        AX_147_200 = frac_in_window(pAX, 147, 200)
        Y_100_147 = frac_in_window(pY, 100, 147)
        Y_147_200 = frac_in_window(pY, 147, 200)

        rows.append({
            "sample": r["sample"],

            "X_100_147": X_100_147,
            "X_147_200": X_147_200,
            "X_ratio": (X_100_147 / X_147_200) if X_147_200 > 0 else np.nan,

            "AutoX_100_147": AX_100_147,
            "AutoX_147_200": AX_147_200,
            "AutoX_ratio": (AX_100_147 / AX_147_200) if AX_147_200 > 0 else np.nan,

            "Y_100_147": Y_100_147,
            "Y_147_200": Y_147_200,
            "Y_ratio": (Y_100_147 / Y_147_200) if Y_147_200 > 0 else np.nan,
        })

    return pd.DataFrame(rows)


# ============================
# Plot boxplot + p-value + caption
# ============================
def make_boxplot(values_left, values_right,
                 label_left: str, label_right: str,
                 ylabel: str, title: str,
                 out_svg: str,
                 pvalue: float = None,
                 q_low: float = 0.01,
                 q_high: float = 0.99,
                 margin_frac: float = 0.12):
    
    values_left = pd.Series(values_left).dropna().astype(float).values
    values_right = pd.Series(values_right).dropna().astype(float).values

    fig, ax = plt.subplots(figsize=(2.75, 4))

    bp = ax.boxplot(
        [values_left, values_right],
        labels=[label_left, label_right],
        patch_artist=True,
        widths=0.6,
        showfliers=False,
        boxprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2),
        medianprops=dict(color="black", linewidth=2.5),
    )

    colors = ["#b2239c", "#5986c1"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14)  # ticks + labels X (Y, Auto+X) plus gros

    # --------- autoscale robuste
    all_vals = np.concatenate([values_left, values_right]) if (len(values_left) + len(values_right)) else np.array([0.0])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        all_vals = np.array([0.0])

    lo = float(np.quantile(all_vals, q_low))
    hi = float(np.quantile(all_vals, q_high))

    # fallback si quantiles identiques
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(all_vals))
        hi = float(np.nanmax(all_vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0

    yr = hi - lo
    pad = margin_frac * yr if yr > 0 else 0.1

    # on garde 0 comme borne basse si tu veux
    y_bottom = max(0.0, lo - 0.05 * yr)
    y_top_data = hi + 0.05 * yr

    # --------- p-value + marge dédiée en haut
    if pvalue is not None:
        ptxt = format_p(pvalue)

        # ligne de comparaison dans la zone "marge"
        y = y_top_data + 0.35 * pad
        ax.plot([1, 1, 2, 2], [y - 0.08 * pad, y, y, y - 0.08 * pad],
                color="black", linewidth=1.2)
        ax.text(1.5, y + 0.12 * pad, ptxt, ha="center", va="bottom", fontsize=10)

        ax.set_ylim(y_bottom, y_top_data + pad)
    else:
        ax.set_ylim(y_bottom, y_top_data)


    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_svg, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_svg}")


if __name__ == "__main__":
    OUT_ROOT = "/data2/USERS/richaud/chrY/distribution_fragments/adalsteisson/"

    print(f"\nReading profiles in: {OUT_ROOT}")
    print("  n chrX:", len(glob.glob(os.path.join(OUT_ROOT, "*_chrX_size.tsv"))))
    print("  n autoX:", len(glob.glob(os.path.join(OUT_ROOT, "*_chr_Auto_X_size.tsv"))))
    print("  n chrY:", len(glob.glob(os.path.join(OUT_ROOT, "*_chrY_size.tsv"))))

    df_paths = build_table(OUT_ROOT)
    df = compute_metrics(df_paths)
    n = int(df.shape[0])

    # --- Wilcoxon paired (Y vs X)
    p_Y_vs_X_100_147   = wilcoxon_paired(df["Y_100_147"], df["X_100_147"])
    p_Y_vs_X_147_200   = wilcoxon_paired(df["Y_147_200"], df["X_147_200"])
    p_Y_vs_X_ratio     = wilcoxon_paired(df["Y_ratio"],   df["X_ratio"])

    # --- Wilcoxon paired (Y vs Auto+X)
    p_Y_vs_AutoX_100_147 = wilcoxon_paired(df["Y_100_147"], df["AutoX_100_147"])
    p_Y_vs_AutoX_147_200 = wilcoxon_paired(df["Y_147_200"], df["AutoX_147_200"])
    p_Y_vs_AutoX_ratio   = wilcoxon_paired(df["Y_ratio"],   df["AutoX_ratio"])

    # --- Boxplots Y vs X
    make_boxplot(df["Y_100_147"], df["X_100_147"], "Y", "X",
                 "Proportion of fragments",
                 f"Prostate Cancer — 100–147 bp (n={n})",
                 os.path.join(OUT_ROOT, "prostate_boxplot_100_147_Y_vs_X.svg"),
                 pvalue=p_Y_vs_X_100_147)

    make_boxplot(df["Y_147_200"], df["X_147_200"], "Y", "X",
                 "Proportion of fragments",
                 f"Prostate Cancer — 147–200 bp (n={n})",
                 os.path.join(OUT_ROOT, "prostate_boxplot_147_200_Y_vs_X.svg"),
                 pvalue=p_Y_vs_X_147_200)

    make_boxplot(df["Y_ratio"], df["X_ratio"], "Y", "X",
                 "Ratio",
                 f"Prostate Cancer — Ratio (100–147 / 147–200) (n={n})",
                 os.path.join(OUT_ROOT, "prostate_boxplot_ratio_Y_vs_X.svg"),
                 pvalue=p_Y_vs_X_ratio)

    # --- Boxplots Y vs Auto+X
    make_boxplot(df["Y_100_147"], df["AutoX_100_147"], "Y", "Auto+X",
                 "Proportion of fragments",
                 f"Prostate Cancer — 100–147 bp (n={n})",
                 os.path.join(OUT_ROOT, "prostate_boxplot_100_147_Y_vs_AutoX.svg"),
                 pvalue=p_Y_vs_AutoX_100_147)

    make_boxplot(df["Y_147_200"], df["AutoX_147_200"], "Y", "Auto+X",
                 "Proportion of fragments",
                 f"Prostate Cancer — 147–200 bp (n={n})",
                 os.path.join(OUT_ROOT, "prostate_boxplot_147_200_Y_vs_AutoX.svg"),
                 pvalue=p_Y_vs_AutoX_147_200)

    make_boxplot(df["Y_ratio"], df["AutoX_ratio"], "Y", "Auto+X",
                 "Ratio",
                 f"Prostate Cancer — Ratio (100–147 / 147–200) (n={n})",
                 os.path.join(OUT_ROOT, "prostate_boxplot_ratio_Y_vs_AutoX.svg"),
                 pvalue=p_Y_vs_AutoX_ratio)

    # --- Sauvegarde p-values
    df_wilcoxon = pd.DataFrame([{
        "n_samples": n,
        "p_Y_vs_X_100_147":     p_Y_vs_X_100_147,
        "p_Y_vs_X_147_200":     p_Y_vs_X_147_200,
        "p_Y_vs_X_ratio":       p_Y_vs_X_ratio,
        "p_Y_vs_AutoX_100_147": p_Y_vs_AutoX_100_147,
        "p_Y_vs_AutoX_147_200": p_Y_vs_AutoX_147_200,
        "p_Y_vs_AutoX_ratio":   p_Y_vs_AutoX_ratio,
    }])
    out_csv = os.path.join(OUT_ROOT, "wilcoxon_pvalues_summary.csv")
    df_wilcoxon.to_csv(out_csv, sep=";", index=False)
    print(f"\nSaved Wilcoxon summary: {out_csv}")
    print("\nTerminé.")
