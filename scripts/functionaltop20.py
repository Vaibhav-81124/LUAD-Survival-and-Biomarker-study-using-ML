import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Paths
data_dir = "../output"
output_dir = "../output"

# Load data
print("Loading merged data for feature extraction...")
data = pd.read_csv(os.path.join(data_dir, "merged_labeled_data.csv"))

# Prepare X and y
X = data.drop(columns=["attrib_name", "status"])
y = data["status"]

# Train-test split (not strictly needed here, but we can train on full data)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RF model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances
feature_names = X.columns
rf_importances = rf_model.feature_importances_

# Create a DataFrame of feature importances
feat_df = pd.DataFrame({
    "Gene": feature_names,
    "Importance": rf_importances
}).sort_values(by="Importance", ascending=False)

# Save top N gene names
top_n = 20
top_genes = feat_df.head(top_n)
top_genes[["Gene"]].to_csv(os.path.join(output_dir, "top_important_genes.txt"), index=False, header=False)

# Save full report
top_genes.to_csv(os.path.join(output_dir, "top_important_genes_with_scores.csv"), index=False)

print(f"\nTop {top_n} important genes saved to output directory.")

