import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import urllib.parse
import os

# === Step 1: Load data ===
data_dir = "../output"
save_dir = "../test"
df = pd.read_csv(os.path.join(data_dir, "merged_labeled_data.csv"))
X = df.drop(columns=["attrib_name", "status"])
y = df["status"]

# === Step 2: Scale & Balance ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

# === Step 3: Train Random Forest ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_bal, y_train_bal)

# === Step 4: Extract Top N Genes ===
importances = rf.feature_importances_
feature_names = X.columns
top_n = 20
top_indices = np.argsort(importances)[-top_n:][::-1]
top_genes = feature_names[top_indices]

# === Step 5: Split Genes by Outcome ===
death_genes = []
survival_genes = []

for gene in top_genes:
    mean_dead = df[df["status"] == 1][gene].mean()
    mean_alive = df[df["status"] == 0][gene].mean()
    if mean_dead > mean_alive:
        death_genes.append(gene)
    else:
        survival_genes.append(gene)

# === Step 6: Save gene lists ===
pd.Series(death_genes).to_csv(os.path.join(save_dir, "death_genes_auto.csv"), index=False)
pd.Series(survival_genes).to_csv(os.path.join(save_dir, "survival_genes_auto.csv"), index=False)

# === Step 7: Generate g:Profiler Links ===
def generate_gprofiler_link(gene_list, title=""):
    base_url = "https://biit.cs.ut.ee/gprofiler/gost"
    gene_str = ",".join(gene_list)
    encoded = urllib.parse.quote(gene_str)
    return f"{base_url}?#query={encoded}&organism=hsapiens&user_threshold=0.05&name={title}"

print("\n🔗 g:Profiler link (Death genes):")
print(generate_gprofiler_link(death_genes, "LUAD_Death_Genes"))

print("\n🔗 g:Profiler link (Survival genes):")
print(generate_gprofiler_link(survival_genes, "LUAD_Survival_Genes"))

