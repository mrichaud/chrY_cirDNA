import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROOT = "/data2/USERS/lruffinel/cfDNA/distribution_fragments/jiang/fragments_tsv"
GROUP_DIRS = {
    "Healthy": os.path.join(ROOT, "Healthy"),
    "Liver cancer": os.path.join(ROOT, "Liver"),
}

OUT_DIR = os.path.join("/data2/USERS/richaud/chrY/distribution_fragments/jiang/comparison_ratio_2")
os.makedirs(OUT_DIR, exist_ok=True)

SHORT_WIN = (0, 147)     # 0..146
LONG_WIN  = (147, 300)   # 147..299
EPS = 1e-12

# -------------------------
# IO helpers
# -------------------------
def load_profile(path):
    arr = np.loadtxt(path).astype(float)
    if arr.shape[0] != 1000:
        raise ValueError(f"Taille inattendue: {path} -> {arr.shape}")
    return arr

def sample_id_from_profile_filename(path):
    base = os.path.basename(path)
    for suf in ("_chrY_size.tsv", "_chr_Auto_X_size.tsv"):
        if base.endswith(suf):
            return base[:-len(suf)]
    return None

def list_pairs_y_autox(dir_tsv):
    y_files = sorted(glob.glob(os.path.join(dir_tsv, "*_chrY_size.tsv")))
    ax_files = sorted(glob.glob(os.path.join(dir_tsv, "*_chr_Auto_X_size.tsv")))

    y_map = {sample_id_from_profile_filename(p): p for p in y_files}
    ax_map = {sample_id_from_profile_filename(p): p for p in ax_files}

    # enlÃ¨ve les clÃ©s None si jamais
    y_map.pop(None, None)
    ax_map.pop(None, None)

    samples = sorted(set(y_map) & set(ax_map))
    return pd.DataFrame([{"sample": s, "path_Y": y_map[s], "path_AutoX": ax_map[s]} for s in samples])

def frac_in_window(profile, start_bp, end_bp):
    return float(np.nansum(profile[start_bp:end_bp]))

# -------------------------
# Scalar computation
# -------------------------
def build_group_scalars(group, dir_tsv):
    pairs = list_pairs_y_autox(dir_tsv)
    if pairs.empty:
        print(f"[{group}] WARNING: aucun Y+AutoX trouvÃ© dans {dir_tsv}")
        return pd.DataFrame(columns=["sample","group","ratio_short","ratio_long","logratio_short","logratio_long"])

    rows = []
    for _, r in pairs.iterrows():
        pY = load_profile(r["path_Y"])
        pAX = load_profile(r["path_AutoX"])

        y_short = frac_in_window(pY, SHORT_WIN[0], SHORT_WIN[1])
        ax_short = frac_in_window(pAX, SHORT_WIN[0], SHORT_WIN[1])
        y_long = frac_in_window(pY, LONG_WIN[0], LONG_WIN[1])
        ax_long = frac_in_window(pAX, LONG_WIN[0], LONG_WIN[1])

        ratio_short = (y_short + EPS) / (ax_short + EPS)
        ratio_long  = (y_long + EPS) / (ax_long + EPS)

        rows.append({
            "sample": r["sample"],
            "group": group,
            "ratio_short": ratio_short,
            "ratio_long": ratio_long,
            "logratio_short": float(np.log(ratio_short)),
            "logratio_long": float(np.log(ratio_long)),
        })

    df = pd.DataFrame(rows)
    print(f"[{group}] n={len(df)}")
    return df

# -------------------------
# KDE plot helpers
# -------------------------
def kde_y(values, x_grid):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return None
    return gaussian_kde(v)(x_grid)

def plot_kde_two_groups(vals_a, label_a, color_a,
                        vals_b, label_b, color_b,
                        x_label, title, out_svg, x_ref=None):
    vals_a = pd.Series(vals_a).dropna().astype(float).values
    vals_b = pd.Series(vals_b).dropna().astype(float).values
    allv = np.concatenate([vals_a, vals_b]) if (len(vals_a) + len(vals_b)) else np.array([0.0])

    q1, q99 = np.quantile(allv, [0.01, 0.99])
    span = q99 - q1
    xmin = q1 - 0.25 * span
    xmax = q99 + 0.25 * span
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        xmin = float(np.min(allv))
        xmax = float(np.max(allv))
        if xmax <= xmin:
            xmin, xmax = -1.0, 1.0

    x_grid = np.linspace(xmin, xmax, 500)

    ya = kde_y(vals_a, x_grid)
    yb = kde_y(vals_b, x_grid)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    if ya is not None:
        ax.plot(x_grid, ya, color=color_a, linewidth=2.5, label=f"{label_a} (n={len(vals_a)})")
        ax.fill_between(x_grid, 0, ya, color=color_a, alpha=0.15)
    else:
        ax.text(0.02, 0.95, f"{label_a}: n<3 (pas de KDE)", transform=ax.transAxes, va="top")

    if yb is not None:
        ax.plot(x_grid, yb, color=color_b, linewidth=2.5, label=f"{label_b} (n={len(vals_b)})")
        ax.fill_between(x_grid, 0, yb, color=color_b, alpha=0.15)
    else:
        ax.text(0.02, 0.88, f"{label_b}: n<3 (pas de KDE)", transform=ax.transAxes, va="top")
    
    if x_ref is not None:
        ax.axvline(x_ref, color="black", linestyle=":", linewidth=1.8)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density (KDE)")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)


    fig.tight_layout()
    fig.savefig(out_svg, dpi=300)
    plt.close(fig)
    print("Saved:", out_svg)

# -------------------------
# Scatter helpers
# -------------------------
def plot_scatter_2groups(df_h, df_c, cancer_name, out_svg, use_log=False):
    x_col = "logratio_long" if use_log else "ratio_long"
    y_col = "logratio_short" if use_log else "ratio_short"
    suffix = "logratio" if use_log else "ratio"

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(df_h[x_col], df_h[y_col], s=35, alpha=0.85, color="#1f77b4",
               label=f"healthy (n={len(df_h)})", edgecolors="none")
    ax.scatter(df_c[x_col], df_c[y_col], s=35, alpha=0.85, color="#d62728",
               label=f"{cancer_name} (n={len(df_c)})", edgecolors="none")

    ax.set_title(f"Short vs Long Y/AutoX ({suffix}) â€” healthy vs {cancer_name} (jiang)", fontweight="bold", size=11.5)
    ax.set_xlabel(f"Long {suffix} (Y/AutoX) on {LONG_WIN[0]}â€“{LONG_WIN[1]} bp")
    ax.set_ylabel(f"Short {suffix} (Y/AutoX) on {SHORT_WIN[0]}â€“{SHORT_WIN[1]} bp")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_svg, dpi=300)
    plt.close(fig)
    print("Saved:", out_svg)

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    # build once
    dfs = [build_group_scalars(g, d) for g, d in GROUP_DIRS.items()]
    df_all = pd.concat(dfs, ignore_index=True)

    # CSV global
    out_csv = os.path.join(OUT_DIR, "scalar_ratios_short_long_ratio_jiang_147bp.csv")
    df_all.to_csv(out_csv, sep=";", index=False)
    print("Saved:", out_csv)

    df_h = df_all[df_all["group"] == "Healthy"].copy()

    # KDEs + scatters per cancer
    for cancer in ["Liver cancer"]:
        df_c = df_all[df_all["group"] == cancer].copy()

        # KDE short ratio & logratio
        plot_kde_two_groups(
            df_h["ratio_short"], "healthy", "#1f77b4",
            df_c["ratio_short"], cancer, "#d62728",
            x_label=f"Ratio Y/AutoX on {SHORT_WIN[0]}â€“{SHORT_WIN[1]} bp",
            title=f"short fragments ({SHORT_WIN[0]}â€“{SHORT_WIN[1]} bp) : healthy vs {cancer} (jiang)",
            out_svg=os.path.join(OUT_DIR, f"ratio_{SHORT_WIN[0]}_{SHORT_WIN[1]}_healthy_vs_{cancer}_jiang.svg"),
            x_ref=None,
        )
        
        # KDE long ratio & logratio
        plot_kde_two_groups(
            df_h["ratio_long"], "healthy", "#1f77b4",
            df_c["ratio_long"], cancer, "#d62728",
            x_label=f"Ratio Y/AutoX on {LONG_WIN[0]}â€“{LONG_WIN[1]} bp",
            title=f"long fragment ({LONG_WIN[0]}â€“{LONG_WIN[1]} bp) : healthy vs {cancer} (jiang)",
            out_svg=os.path.join(OUT_DIR, f"ratio_{LONG_WIN[0]}_{LONG_WIN[1]}_healthy_vs_{cancer}_jiang.svg"),
            x_ref=None,
        )

        # scatters 2 groupes (ratio + logratio)
        plot_scatter_2groups(
            df_h, df_c, cancer,
            os.path.join(OUT_DIR, f"scatter_ratio_short_vs_long_healthy_vs_{cancer}_147bp_jiang.svg"),
            use_log=False
        )


    print("Done. Output:", OUT_DIR)
