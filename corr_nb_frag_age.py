from concurrent.futures import ProcessPoolExecutor
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd


def get_y_count(indiv_path: str) -> int:
    """Reads TSV file in fast chunks and returns total Chr Y reads with score >= 30."""
    try:
        count = 0
        reader = pd.read_csv(
            indiv_path,
            sep="\t",
            header=None,
            usecols=[0, 3],
            names=["chr", "score"],
            dtype={"chr": "category", "score": "int16"},
            engine="c",
            chunksize=500_000,
        )

        for chunk in reader:
            clean_chr = chunk["chr"].str.removeprefix("chr")
            mask = (clean_chr == "Y") & (chunk["score"] >= 30)
            count += mask.sum()

        return count

    except Exception as e:
        print(f"Error processing {indiv_path}: {e}")
        return 0


if __name__ == "__main__":
    # 1. Read metadata
    annot_path = "/data2/USERS/lruffinel/cfDNA/distribution_fragments/cristiano/cristiano_patients_annotation.csv"
    annot = pd.read_csv(annot_path, sep=";")

    # Filter male annotations and drop duplicates
    male_annot = annot[annot["Sexe"] == "M"].copy()
    male_annot = male_annot.drop_duplicates(subset=["ID_FinaleDB"])

    # Map ID -> Age for quick, safe lookup
    age_lookup = dict(zip(male_annot["ID_FinaleDB"], male_annot["Age"]))

    # 2. Get and filter input files
    all_indivs = glob.glob(
        "/data2/USERS/BIOINFO/databases/FinaleDB/cristiano-data-2019/*.tsv"
    )

    # Build structured list of (filepath, sample_id, age) to guarantee matching alignment
    ordered_samples = []
    for filepath in all_indivs:
        sample_id = os.path.basename(filepath).split(".hg38")[0]
        if sample_id in age_lookup:
            ordered_samples.append({
                "filepath": filepath,
                "sample_id": sample_id,
                "age": age_lookup[sample_id],
            })

    print(
        f"Processing {len(ordered_samples)} male samples across {os.cpu_count()} cores..."
    )

    # Extract target file paths in fixed order
    target_paths = [s["filepath"] for s in ordered_samples]
    print(target_paths)

    # 3. Parallel Execution with executor.map (Guarantees Output Order == Input Order)
    with ProcessPoolExecutor(max_workers=20) as executor:
        counts = list(executor.map(get_y_count, target_paths))

    # 4. Construct DataFrame from strict 1:1 index alignment
    df_results = pd.DataFrame({
        "Sample_ID": [s["sample_id"] for s in ordered_samples],
        "Age": [s["age"] for s in ordered_samples],
        "Count": counts,
    })

    # Sort strictly by Age for plotting
    df_results = df_results.sort_values(by="Age").reset_index(drop=True)

    # 5. Plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(
        df_results["Age"],
        df_results["Count"],
        alpha=0.7,
        edgecolors="none",
        c="royalblue",
    )
    plt.xlabel("Age")
    plt.ylabel("Chr Y Fragment Count (MAPQ >= 30)")
    plt.title("Chromosome Y Read Counts vs Age (Male Patients)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("counts_ages.svg")
    plt.close()

    print("Success! Processed and saved correctly aligned data to counts_ages.svg")