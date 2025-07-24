import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your enrichment results
file_path = "../output/gProfiler_hsapiens_11-04-2025_12-30-47__intersections.csv"
df = pd.read_csv(file_path)

# Select top N terms by lowest adjusted p-value
top_n = 10
df_top = df.sort_values("adjusted_p_value").head(top_n)

# Set plot style
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# Barplot with log10 significance
sns.barplot(
    y="term_name",
    x="negative_log10_of_adjusted_p_value",
    data=df_top,
    palette="viridis"
)

# Plot details
plt.xlabel("-log10(Adjusted P-value)")
plt.ylabel("Enriched Term")
plt.title(f"Top {top_n} Enriched Terms from g:Profiler")
plt.tight_layout()

# Save to output
output_path = "../output/gprofiler_barplot.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"✅ Bar plot saved to: {output_path}")

