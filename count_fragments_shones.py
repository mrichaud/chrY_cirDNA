import re
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --------- paramÃ¨tres ---------
BED_DIR = Path("/data2/USERS/lruffinel/cfDNA/distribution_fragments/Shones/fragments_bed/")
PATTERN = "ActivatedNucleosomes-*.bed"

CHR_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

def count_bed_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="latin-1", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("track") or line.startswith("browser"):
                continue
            n += 1
    return n

def extract_chr(filename):
    m = re.search(r"\b(chr(?:[1-9]|1[0-9]|2[0-2]|X|Y))\b", filename)
    return m.group(1) if m else None 

def comma_fmt(x, pos):
    return f"{int(x):,}".replace(",", " ")

def plot_fragment_counts(df):
    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(6,6))

    sns.barplot(
        data=df,
        x="chr",
        y="n_fragments",
        order=CHR_ORDER,
        color="#99237F",
        edgecolor="black",
        linewidth=0.6,
        ax=ax,
    )

    ax.set_title("Distribution des fragments par chromosome prÃ©sents dans les lignÃ©es cellulaires LTCD4+ ActivÃ©es (Shones)", pad=10, fontweight="bold", loc="left")
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Nombre de fragments")
    ax.yaxis.set_major_formatter(FuncFormatter(comma_fmt))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_max = df["n_fragments"].max()
    ax.set_ylim(0, y_max * 1.12)

    fig.tight_layout()
    fig.savefig("/data2/USERS/richaud/chrY/distribution_fragments/Shones/fragment_counts_by_chr_barplot.svg", dpi=300)

if __name__ == "__main__":
    files = sorted(BED_DIR.glob(PATTERN))
    if not files:
        raise SystemExit(f"Aucun fichier trouvÃ©: {BED_DIR}/{PATTERN}")

    rows = []
    total_files = len(files)

    for i, fp in enumerate(files, start=1):
        chr_name = extract_chr(fp.name)
        if chr_name is None:
            print(f"[{i}/{total_files}] SKIP (nom non reconnu): {fp.name}")
            continue

        print(f"[{i}/{total_files}] Counting {chr_name} from {fp.name} ...", flush=True)
        n = count_bed_lines(fp)
        print(f"[{i}/{total_files}] Done {chr_name}: {n} fragments", flush=True)

        rows.append({"chr": chr_name, "n_fragments": n, "file": fp.name})

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Aucun chromosome parsÃ© depuis les noms de fichiers (regex).")

    df["chr"] = pd.Categorical(df["chr"], categories=CHR_ORDER, ordered=True)
    df = df.sort_values("chr")

    df[["chr", "n_fragments"]].to_csv("/data2/USERS/richaud/chrY/distribution_fragments/Shones/fragment_counts_by_chr.tsv", sep="\t", index=False)

    plot_fragment_counts(df)

    print("terminÃ©")
