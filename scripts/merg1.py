import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set data directory
data_dir = "../output"
mutation_path = os.path.join(data_dir, "mutation_cleaned.csv")
rnaseq_path = os.path.join(data_dir, "rnaseq_cleaned.csv")
clinical_path = os.path.join(data_dir, "clinical_cleaned.csv")

# Load and transpose mutation & rnaseq
mutation_df = pd.read_csv(mutation_path, index_col=0).transpose()
rnaseq_df = pd.read_csv(rnaseq_path, index_col=0).transpose()

# Reset index to make 'attrib_name' a column
mutation_df.reset_index(inplace=True)
rnaseq_df.reset_index(inplace=True)

# Rename index column to match across datasets
mutation_df.rename(columns={"index": "attrib_name"}, inplace=True)
rnaseq_df.rename(columns={"index": "attrib_name"}, inplace=True)

# Merge omics data on sample ID
merged_df = pd.merge(mutation_df, rnaseq_df, on="attrib_name", how="inner")

# Load and transpose clinical data
clinical_df = pd.read_csv(clinical_path, index_col=0).transpose().reset_index()
clinical_df.rename(columns={"index": "attrib_name"}, inplace=True)

# Merge clinical status
if "status" not in clinical_df.columns:
    raise ValueError("Status column not found in transposed clinical data!")

merged_df = merged_df.merge(clinical_df[["attrib_name", "status"]], on="attrib_name", how="inner")

# Drop rows with missing labels
merged_df = merged_df.dropna(subset=["status"])

# Save merged data
output_path = os.path.join(data_dir, "merged_labeled_data.csv")
merged_df.to_csv(output_path, index=False)
print(f"Saved merged data to {output_path}")

# Plot label distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="status", data=merged_df)
plt.title("Survival Label Distribution (status)")
plt.xlabel("Status (0 = Alive, 1 = Deceased)")
plt.ylabel("Count")
plt.tight_layout()
plot_path = os.path.join(data_dir, "label_distribution.png")
plt.savefig(plot_path)
print(f"Saved label distribution plot to {plot_path}")
plt.show()
