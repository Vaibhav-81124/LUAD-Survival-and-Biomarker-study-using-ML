import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Increase recursion depth for dendrograms
sys.setrecursionlimit(10000)

# Paths
data_dir = "../output"
output_dir = "../output"

# Load data
print("Loading merged labeled data...")
data = pd.read_csv(os.path.join(data_dir, "merged_labeled_data.csv"))
X = data.drop(columns=["attrib_name", "status"])
y = data["status"]

# Select top 100 most variable features
top_100 = X.var().sort_values(ascending=False).head(100).index
X_top = X[top_100].copy()
X_top["status"] = y

# Sort samples by class
X_top = X_top.sort_values("status")

# Drop labels for heatmap
X_top_plot = X_top.drop(columns=["status"])

# Generate clustered heatmap (uses SciPy by default)
print("Generating clustered heatmap using SciPy...")
sns.set(style="white")
clustermap = sns.clustermap(
    X_top_plot,
    cmap="vlag",
    figsize=(14, 10),
    yticklabels=False,
    xticklabels=True,
    row_cluster=True,
    col_cluster=True,   # Enable column clustering
    metric="euclidean", # Distance metric
    method="average"    # Linkage method
)

clustermap.fig.suptitle("Clustered Heatmap of Top 100 Features", fontsize=16)
clustermap.ax_heatmap.set_xlabel("Gene Features")
clustermap.ax_heatmap.set_ylabel("Samples")

# Save the plot
heatmap_path = os.path.join(output_dir, "clustered_heatmap_top100_scipy.png")
clustermap.savefig(heatmap_path)
print(f"Heatmap saved to {heatmap_path}")

