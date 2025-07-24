import os
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Load data
clinical = pd.read_csv("../output/clinical_cleaned.csv", header=None)
rnaseq = pd.read_csv("../output/rnaseq_cleaned.csv", header=None, low_memory=False)
top_genes = pd.read_csv("../test/survival_genes_auto.csv")  # Not used here but might be for next step
result_dir="../test"

# Prepare clinical
clinical = clinical.set_index(0).transpose()

# Prepare RNAseq
rnaseq = rnaseq.set_index(0).transpose()

# Merge on 'attrib_name'
if 'attrib_name' not in clinical.columns or 'attrib_name' not in rnaseq.columns:
    raise KeyError("Both clinical and rnaseq DataFrames must contain an 'attrib_name' column after transpose.")

merged = pd.merge(clinical, rnaseq, on='attrib_name', how='inner')

# Set 'attrib_name' as index (optional)
merged = merged.set_index('attrib_name')

# Convert survival-related columns to numeric
merged['overall_survival'] = pd.to_numeric(merged['overall_survival'], errors='coerce')
merged['status'] = pd.to_numeric(merged['status'], errors='coerce')

# Drop rows with missing time or event data
merged = merged.dropna(subset=['overall_survival', 'status'])

# Extract time and event
T = merged['overall_survival']
E = merged['status']

# Plot Kaplan-Meier survival curve
kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 6))
kmf.fit(durations=T, event_observed=E, label="Overall Survival")
kmf.plot_survival_function(ci_show=False)

plt.title('Kaplan-Meier Survival Curve')
plt.xlabel('Days')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.tight_layout()

# Save plot
plot_path = os.path.join(result_dir, 'survival_plot.png')
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"✅ Plot saved to {plot_path}")

