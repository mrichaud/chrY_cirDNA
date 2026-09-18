import os
import glob
from multiprocessing import Pool
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Canonical chromosome order
CHR_LIST = [str(i) for i in range(1, 23)] + ["X", "Y"]
CHR_ORDER_MAP = {chrom: idx for idx, chrom in enumerate(CHR_LIST)}

def size_distrib(indiv):
    try:
        # Read only required columns (col 0: chr, col 3: score)
        df = pd.read_csv(
            indiv, 
            sep="\t", 
            header=None, 
            usecols=[0, 3], 
            names=["chr", "score"],
            dtype={"chr": str, "score": int}
        )

        # Filter high-quality scores
        df = df[df["score"] >= 30]

        # Clean chromosome names ('chr1' -> '1')
        df["chr"] = df["chr"].str.replace("chr", "", regex=False)
        
        # Keep only chromosomes in target list
        df = df[df["chr"].isin(CHR_ORDER_MAP)]

        # Vectorized fragment counting per chromosome
        counts = df["chr"].value_counts()

        # Build count vector (length 24) in the correct chromosome order
        frag_counter = [counts.get(chrom, 0) for chrom in CHR_LIST]

        return frag_counter if sum(frag_counter) > 0 else None

    except Exception as e:
        print(f"Error processing {indiv}: {e}")
        return None


if __name__ == "__main__":
    # 1. Read annotations ONCE in the main process
    annot_path = "/data2/USERS/lruffinel/cfDNA/distribution_fragments/cristiano/cristiano_patients_annotation.csv"
    annot = pd.read_csv(annot_path, sep=";")
    males_set = set(annot[annot["Sexe"] == "M"]["ID_FinaleDB"].to_list())

    # 2. Get all target files
    all_indivs = glob.glob("/data2/USERS/BIOINFO/databases/FinaleDB/cristiano-data-2019/*.tsv")

    # 3. Pre-filter files to process ONLY male patients before parallelizing
    target_indivs = [
        filepath for filepath in all_indivs
        if os.path.basename(filepath).split(".hg38")[0] in males_set
    ]

    print(f"Found {len(target_indivs)} male samples to process out of {len(all_indivs)} files.")

    # 4. Run parallel processing cleanly (adjust processes as needed, e.g., os.cpu_count())
    num_processes = min(10, len(target_indivs)) if target_indivs else 1
    with Pool(processes=num_processes) as p:
        res = p.map(size_distrib, target_indivs)

    # Filter out None results
    indivs_counts = [c for c in res if c is not None]

    if indivs_counts:
        # Chromosome sizes (hg38 lengths, Chr 5 corrected)
        chr_sizes = [
            248956422, 242193529, 198295559, 190214555, 181538259, 170805979,
            159345973, 145138636, 138394717, 133797422, 135086622, 133275309,
            114364328, 107043718, 101991189, 90338345,  83257441,  80373285,
            58617616,  64444167,  46709983,  50818468,  156040895, 57227415
        ]

        df = pd.DataFrame(indivs_counts, columns=CHR_LIST)
        df = df.div(chr_sizes, axis=1)

        # Plot setup
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df)
        plt.xlabel("Chromosome")
        plt.ylabel("Normalized Fragment Count")
        plt.tight_layout()
        plt.savefig("frag_counts_boxplots.svg", dpi=300, format="svg", bbox_inches="tight")
        plt.close()
        print("Plot saved successfully.")
    else:
        print("No valid data collected.")