import os
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Load data
clinical = pd.read_csv("../output/clinical_cleaned.csv", header=None)
rnaseq = pd.read_csv("../output/rnaseq_cleaned.csv", header=None, low_memory=False)

# Transpose clinical and RNAseq dataframes (since rows are samples)
clinical = clinical.transpose()
rnaseq = rnaseq.transpose()

# Rename columns (the first row of each dataframe should be the headers)
clinical.columns = clinical.iloc[0]
clinical = clinical.drop(0)  # Drop the first row which contains the column names
rnaseq.columns = rnaseq.iloc[0]
rnaseq = rnaseq.drop(0)  # Drop the first row which contains the column names

# Ensure 'attrib_name' is present in both dataframes
if 'attrib_name' not in clinical.columns:
    raise KeyError("Clinical dataframe must contain 'attrib_name' column.")
if 'attrib_name' not in rnaseq.columns:
    raise KeyError("RNAseq dataframe must contain 'attrib_name' column.")

# Merge clinical and RNAseq data on 'attrib_name'
merged = pd.merge(clinical, rnaseq, on='attrib_name', how='inner')

# Make sure survival columns are numeric
merged['overall_survival'] = pd.to_numeric(merged['overall_survival'], errors='coerce')
merged['status'] = pd.to_numeric(merged['status'], errors='coerce')

# Drop missing survival data
merged = merged.dropna(subset=['overall_survival', 'status'])

# Create output directory for plots
output_dir = "../test"

# Time and event columns
T = merged['overall_survival']
E = merged['status']

# Initialize Kaplan-Meier fitter
kmf = KaplanMeierFitter()

# Loop through each filtered gene in the CSV
filtered_genes = pd.read_csv("../test/survival_genes_auto.csv")  # This CSV should have a column like "Gene"

for gene in filtered_genes["Gene"]:  # Assuming column name is "Gene" in your CSV
    if gene not in merged.columns:
        print(f"⚠️  Skipping {gene} - not found in RNAseq data")
        continue

    # Ensure the gene expression column is numeric
    merged[gene] = pd.to_numeric(merged[gene], errors='coerce')
    merged = merged.dropna(subset=[gene])  # Drop rows with NaN values in the gene column

    # Group by median expression (high vs low)
    median_expr = merged[gene].median()
    merged['group'] = merged[gene].apply(lambda x: "High" if x > median_expr else "Low")

    # Plot KM curve
    plt.figure(figsize=(8, 6))

    for group in ['High', 'Low']:
        mask = merged['group'] == group
        kmf.fit(durations=T[mask], event_observed=E[mask], label=group)
        kmf.plot_survival_function(ci_show=False)

    plt.title(f"Survival Curve for {gene}")
    plt.xlabel("Days")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.legend(title="Expression Level")
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, f"{gene}_survival.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Log-rank test (optional)
    mask_high = merged["group"] == "High"
    mask_low = merged["group"] == "Low"
    results = logrank_test(T[mask_high], T[mask_low], E[mask_high], E[mask_low])
    print(f"{gene}: log-rank p-value = {results.p_value:.4f}")

print("\n✅ Survival analysis complete!")
print(f"Plots saved in: {output_dir}")

