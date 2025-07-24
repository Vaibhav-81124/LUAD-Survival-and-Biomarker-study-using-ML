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
