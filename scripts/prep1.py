import pandas as pd
import os

# Set paths
data_dir = "../data"
output_dir = "../output"

mutation_path = os.path.join(data_dir, "mutation.cbt")
rnaseq_path = os.path.join(data_dir, "RNA.cct")
clinical_path = os.path.join(data_dir, "clinical.tsi")

# Create output dir if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 1. Load datasets
print("🔍 Loading mutation data...")
mutation_df = pd.read_csv(mutation_path, sep='\t')

print("🔍 Loading RNA-seq data...")
rnaseq_df = pd.read_csv(rnaseq_path, sep='\t')

print("🔍 Loading clinical data...")
clinical_df = pd.read_csv(clinical_path, sep='\t')

# 2. Preview data
print("✅ Mutation Data:", mutation_df.shape)
print("✅ RNA-Seq Data:", rnaseq_df.shape)
print("✅ Clinical Data:", clinical_df.shape)

# 3. Clean RNA-seq data (example: drop nulls, genes as rows if needed)
rnaseq_df.dropna(inplace=True)

# 4. Optional: clean mutation or clinical data based on use case
# mutation_df = mutation_df[mutation_df['Tumor_Sample_Barcode'].notna()]

# 5. Save cleaned versions
mutation_df.to_csv(os.path.join(output_dir, "mutation_cleaned.csv"), index=False)
rnaseq_df.to_csv(os.path.join(output_dir, "rnaseq_cleaned.csv"), index=False)
clinical_df.to_csv(os.path.join(output_dir, "clinical_cleaned.csv"), index=False)

print("✅ Preprocessing complete. Cleaned data saved in /output")

