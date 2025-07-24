import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Step 1: Load data ===
data_dir = "../output"
save_dir = "../test"
df = pd.read_csv(os.path.join(data_dir, "merged_labeled_data.csv"))

# Read top genes from functional analysis output
death_genes = pd.read_csv(os.path.join(save_dir, "death_genes_auto.csv"), header=None)[0].tolist()
survival_genes = pd.read_csv(os.path.join(save_dir, "survival_genes_auto.csv"), header=None)[0].tolist()

# === Count how many patients have each gene non-zero ===
gene_labels = []
gene_counts = []

for gene in death_genes:
    if gene in df.columns:
        gene_labels.append(f"{gene} (death)")
        gene_counts.append((df[gene] != 0).sum())

for gene in survival_genes:
    if gene in df.columns:
        gene_labels.append(f"{gene} (survival)")
        gene_counts.append((df[gene] != 0).sum())

# === Plotting ===
plt.figure(figsize=(12, 6))
colors = ["#d62728" if "(death)" in g else "#2ca02c" for g in gene_labels]
sns.barplot(x=gene_labels, y=gene_counts, palette=colors)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Number of Patients")
plt.xlabel("Gene (Death or Survival)")
plt.title("Number of Patients with Each Top Gene")
plt.tight_layout()

# Save the plot
plot_path = os.path.join(save_dir, "gene_patient_barplot.png")
plt.savefig(plot_path)
plt.show()

print(f"✅ Plot saved to: {plot_path}")

