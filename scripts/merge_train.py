import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
data_dir = "../output"
output_dir = "../output"

# Load mutation, RNA-seq, and clinical data
print("Loading data...")
mutation_df = pd.read_csv(os.path.join(data_dir, "mutation_cleaned.csv"))
rnaseq_df = pd.read_csv(os.path.join(data_dir, "rnaseq_cleaned.csv"))
clinical_df = pd.read_csv(os.path.join(data_dir, "clinical_cleaned.csv"), index_col=0)

# Merge mutation and RNA-seq data
print("Merging mutation and RNA-seq data...")
merged_df = pd.merge(mutation_df, rnaseq_df, on="attrib_name", how="inner")

# Process clinical data: transpose and clean
print("Processing clinical data...")
clinical_df = clinical_df.transpose().reset_index()
clinical_df = clinical_df.rename(columns={"index": "attrib_name"})
clinical_df["status"] = pd.to_numeric(clinical_df["status"], errors="coerce")

# Merge with clinical status
print("Merging with clinical status...")
merged_df = merged_df.merge(clinical_df[["attrib_name", "status"]], on="attrib_name", how="inner")
merged_df = merged_df.dropna(subset=["status"])

# Save the merged data
output_path = os.path.join(output_dir, "merged_labeled_data.csv")
merged_df.to_csv(output_path, index=False)
print(f"✅ Saved merged labeled data to: {output_path}")

# Plot and save label distribution
sns.countplot(data=merged_df, x="status")
plt.title("Survival Label Distribution (status)")
plt.xlabel("Status (0 = Alive, 1 = Deceased)")
plt.ylabel("Count")
plt.tight_layout()

plot_path = os.path.join(output_dir, "label_distribution.png")
plt.savefig(plot_path)
plt.close()
print(f"📊 Saved label distribution plot to: {plot_path}")


