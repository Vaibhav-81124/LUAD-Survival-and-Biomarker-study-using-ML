import pandas as pd
import os
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# === Load Data ===
data_dir = "../output"
save_dir = "../test"
omics_df = pd.read_csv(os.path.join(data_dir, "merged_labeled_data.csv"))
clinical_raw = pd.read_csv(os.path.join(data_dir, "clinical_cleaned.csv"))
survival_genes_path = os.path.join(save_dir, "survival_genes_auto.csv")

# === Properly Process Clinical Data ===
# Transpose
clinical_df = clinical_raw.transpose()

# Set the first row as header (correct column names)
clinical_df.columns = clinical_df.iloc[0]
clinical_df = clinical_df.drop(clinical_df.index[0]).reset_index().rename(columns={"index": "attrib_name"})

print("Columns after transpose:", clinical_df.columns.tolist())

# === Merge Omics and Clinical Data ===
merged_df = pd.merge(
    omics_df,
    clinical_df[["attrib_name", "overall_survival", "status"]],
    on="attrib_name",
    how="inner"
)

# === Load survival genes ===
survival_genes = pd.read_csv(survival_genes_path, header=None).squeeze().tolist()

# === Generate Kaplan-Meier plots ===
for gene in survival_genes:
    if gene not in merged_df.columns:
        print(f"Skipping gene {gene} (not found in data)")
        continue

    try:
        df_gene = merged_df[["overall_survival", "status", gene]].dropna()
        
        # Split into groups based on gene presence
        mask_present = df_gene[gene] != 0
        mask_absent = df_gene[gene] == 0

        kmf_present = KaplanMeierFitter()
        kmf_absent = KaplanMeierFitter()

        plt.figure()
        kmf_present.fit(durations=df_gene[mask_present]["overall_survival"], 
                        event_observed=df_gene[mask_present]["status"],
                        label=f"{gene} Present")
        kmf_absent.fit(durations=df_gene[mask_absent]["overall_survival"], 
                       event_observed=df_gene[mask_absent]["status"],
                       label=f"{gene} Absent")

        # Plot curves
        kmf_present.plot_survival_function()
        kmf_absent.plot_survival_function()
        plt.title(f"Kaplan-Meier Curve for {gene}")
        plt.xlabel("Survival Time (days)")
        plt.ylabel("Survival Probability")
        plt.grid(True)
        plt.tight_layout()

        # Save each plot
        plt.savefig(os.path.join(save_dir, f"km_curve_{gene}.png"))
        plt.close()

    except Exception as e:
        print(f"Error plotting {gene}: {e}")

print("✅ Kaplan-Meier plots saved successfully!")

