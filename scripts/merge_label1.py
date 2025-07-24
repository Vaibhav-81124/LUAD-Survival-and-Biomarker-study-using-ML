import os
import pandas as pd

# Set paths
output_dir = "../output"
data_dir = "../data"
mutation_path = os.path.join(output_dir, "mutation_cleaned.csv")
rnaseq_path = os.path.join(output_dir, "rnaseq_cleaned.csv")
driver_path = os.path.join(data_dir, "luad_driver_genes.csv")  # Driver gene list

# Load datasets
print("Loading cleaned data...")
mutation_df = pd.read_csv(mutation_path)
rnaseq_df = pd.read_csv(rnaseq_path)
driver_df = pd.read_csv(driver_path)

# Standardize column names
print("Standardizing column names...")
mutation_df.rename(columns=lambda x: x.strip().lower(), inplace=True)
rnaseq_df.rename(columns=lambda x: x.strip().lower(), inplace=True)
driver_df.rename(columns=lambda x: x.strip().lower(), inplace=True)

# Rename 'arrtib_name' to 'gene' for merging
mutation_df.rename(columns={"arrtib_name": "gene"}, inplace=True)
rnaseq_df.rename(columns={"arrtib_name": "gene"}, inplace=True)
# After renaming columns
print("\nMutation columns:", mutation_df.columns.tolist())
print("RNA-seq columns:", rnaseq_df.columns.tolist())
print("Driver gene columns:", driver_df.columns.tolist())


# Merge mutation and RNA-seq data on 'gene'
print("Merging mutation and RNA-seq data...")
merged_df = pd.merge(mutation_df, rnaseq_df, on="gene", how="outer")

# Labeling: 1 if gene is a known driver gene, else 0
print("Applying driver gene labels...")
merged_df["gene"] = merged_df["gene"].astype(str).str.strip().str.upper()
driver_genes = driver_df["gene"].astype(str).str.strip().str.upper().unique().tolist()

merged_df["Label"] = merged_df["gene"].apply(lambda x: 1 if x in driver_genes else 0)

# Save merged and labeled data
output_path = os.path.join(output_dir, "merged_labeled_data1.csv")
merged_df.to_csv(output_path, index=False)

print(f"\n✅ Merged and labeled data saved to: {output_path}")

