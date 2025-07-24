import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set up path
data_dir = "../output"

# Load merged and labeled data
data_path = os.path.join(data_dir, "merged_labeled_data.csv")
df = pd.read_csv(data_path)

# Drop rows with missing status values
df = df.dropna(subset=["status"])

# Separate features and label
features = df.drop(columns=["attrib_name", "status"])
labels = df["status"]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 1. Plot class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=labels)
plt.title("Label Distribution (Survival Status)")
plt.xlabel("Survival Status (0 = Deceased, 1 = Alive)")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "plot_class_distribution.png"))
plt.close()

# 2. PCA 2D scatter plot
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])
pca_df["status"] = labels.values

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="status", palette="Set1", s=60)
plt.title("PCA of Samples Colored by Survival Status")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "plot_pca_scatter.png"))
plt.close()

# 3. Correlation heatmap of top 20 variable features
variances = pd.DataFrame(scaled_features, columns=features.columns).var().sort_values(ascending=False)
top_features = features[variances.head(20).index]

plt.figure(figsize=(12, 10))
sns.heatmap(top_features.corr(), cmap="coolwarm", annot=False, fmt=".2f", square=True,
            cbar_kws={"label": "Correlation Coefficient"})
plt.title("Correlation Heatmap of Top 20 Features")
plt.xlabel("Features")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "plot_top20_corr_heatmap.png"))
plt.close()

