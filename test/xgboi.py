import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# === Paths ===
data_dir = "../output"  # Load data from output directory
result_dir = "../test"   # Save results to test directory
data_path = os.path.join(data_dir, "merged_labeled_data.csv")

# === Load Data ===
data = pd.read_csv(data_path)
X = data.drop(columns=["attrib_name", "status"])
y = data["status"]

# === Scale Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train-test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Apply SMOTE ===
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# === Define and Train XGBoost Classifier ===
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_sm, y_train_sm)

# === Predictions ===
y_pred = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)[:, 1]

# === Evaluation ===
print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "xgboost_confusion_matrix.png"))
plt.show()

# === ROC Curve ===
RocCurveDisplay.from_estimator(xgb, X_test, y_test)
plt.title("XGBoost ROC Curve")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "xgboost_roc_curve.png"))
plt.show()

print("\n✅ XGBoost model results saved to test directory.")
