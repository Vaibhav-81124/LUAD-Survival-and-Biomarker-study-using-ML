import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set path
data_dir = "../output"
merged_file = os.path.join(data_dir, "merged_labeled_data.csv")
df = pd.read_csv(merged_file)

# Separate features and label
X = df.drop(columns=["attrib_name", "status"])
y = df["status"]

# Standardize features
X_scaled = StandardScaler().fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Get feature names
feature_names = X.columns

# Get loadings for PC1 and PC2
loadings = pd.DataFrame(pca.components_.T,
                        columns=['PC1', 'PC2'],
                        index=feature_names)

top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(10)
top_pc2 = loadings["PC2"].abs().sort_values(ascending=False).head(10)

# Plot combined figure
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# PCA Scatter plot
scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6, edgecolor='k')
axes[0].set_title("PCA of Samples by Survival Status")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")
legend_labels = ["Alive (0)", "Deceased (1)"]
legend = axes[0].legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Status")

# Top features PC1
axes[1].barh(top_pc1.index[::-1], top_pc1.values[::-1], color='skyblue')
axes[1].set_title("Top 10 Features Contributing to PC1")
axes[1].set_xlabel("Absolute Contribution")
axes[1].set_ylabel("Feature")

# Top features PC2
axes[2].barh(top_pc2.index[::-1], top_pc2.values[::-1], color='salmon')
axes[2].set_title("Top 10 Features Contributing to PC2")
axes[2].set_xlabel("Absolute Contribution")
axes[2].set_ylabel("Feature")

plt.tight_layout()
combined_path = os.path.join(data_dir, "pca_combined_analysis.png")
plt.savefig(combined_path)
plt.close()

print(f"✅ PCA combined figure saved to: {combined_path}")

