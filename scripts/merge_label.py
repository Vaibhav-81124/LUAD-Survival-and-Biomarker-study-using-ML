import pandas as pd
import os

# Paths
output_dir = "../output/"
mutation_file = os.path.join(output_dir, "mutation_cleaned.csv")
rnaseq_file = os.path.join(output_dir, "rnaseq_cleaned.csv")
clinical_file = os.path.join(output_dir, "clinical_cleaned.csv")
driver_file = os.path.join(output_dir, "luad_driver_genes.csv")
merged_file = os.path.join(output_dir, "merged_labeled_data.csv")

# Load DataFrames
mutation_df = pd.read_csv(mutation_file)
rnaseq_df = pd.read_csv(rnaseq_file)
clinical_df = pd.read_csv(clinical_file)
drivers_df = pd.read_csv(driver_file)

# Print column names to debug
print("\n🔍 Columns in mutation data:", mutation_df.columns.tolist())
print("🔍 Columns in RNA-seq data:", rnaseq_df.columns.tolist())
print("🔍 Columns in driver gene data:", drivers_df.columns.tolist())

# Use 'attrib_name' as the gene column
gene_column = 'attrib_name'

# Make gene names uppercase for consistency
mutation_df[gene_column] = mutation_df[gene_column].str.upper()
rnaseq_df[gene_column] = rnaseq_df[gene_column].str.upper()
drivers_df['gene'] = drivers_df['gene'].str.upper()

# Merge mutation and RNA-seq data on 'attrib_name'
merged_df = pd.merge(mutation_df, rnaseq_df, on=gene_column, how="outer")

# Label as driver (1) or non-driver (0)
merged_df["label"] = merged_df[gene_column].isin(drivers_df["gene"]).astype(int)

# Save output
merged_df.to_csv(merged_file, index=False)
print(f"\n✅ Merged and labeled data saved to: {merged_file}")
