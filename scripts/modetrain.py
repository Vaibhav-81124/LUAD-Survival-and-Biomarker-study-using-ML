# scripts/train_models.py

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel

# === Set paths ===
data_dir = "../output"
output_dir = "../output"

# === Load the merged and labeled dataset ===
data_path = os.path.join(data_dir, "merged_labeled_data.csv")
print("Loading data from:", data_path)
df = pd.read_csv(data_path)

# === Preprocess ===
X = df.drop(columns=["attrib_name", "status"])  # Drop ID and label columns
y = df["status"]  # 0 = Alive, 1 = Dead

# Handle missing values
X.fillna(0, inplace=True)

# === Feature Selection using Random Forest ===
print("Selecting top features using Random Forest...")
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X, y)

model = SelectFromModel(rf_selector, prefit=True, max_features=50, threshold=-np.inf)
X_selected = model.transform(X)
selected_features = X.columns[model.get_support()]
pd.Series(selected_features).to_csv(os.path.join(output_dir, "selected_features.csv"), index=False)

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# === Train Random Forest ===
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# === Train SVM ===
svm_model = SVC(kernel="linear", probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

# === Evaluation & Plotting ===
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Classification Report:")
    print(classification_report(y_true, y_pred))

    # Save report
    with open(os.path.join(output_dir, f"{name.lower()}_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(output_dir, f"{name.lower()}_confusion_matrix.png"))
    plt.close()

# Evaluate both models
evaluate_model("RandomForest", y_test, rf_preds)
evaluate_model("SVM", y_test, svm_preds)

print("✅ Training and evaluation complete. Outputs saved in output/")

