import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             roc_curve, auc)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set paths
data_dir = "../output"
output_dir = "../output"

# Load data
print("Loading labeled dataset...")
data = pd.read_csv(os.path.join(data_dir, "merged_labeled_data.csv"))

# Drop non-feature columns and set up X and y
X = data.drop(columns=["attrib_name", "status"])
y = data["status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_score_rf = rf_model.predict_proba(X_test)[:, 1]

# SVM
print("Training SVM...")
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
y_score_svm = svm_model.predict_proba(X_test)[:, 1]

# Evaluate and print metrics
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(f"Classification Report:\n{classification_report(y_true, y_pred)}")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name.lower()}_confusion_matrix.png"))
    plt.close()

evaluate_model("RandomForest", y_test, y_pred_rf)
evaluate_model("SVM", y_test, y_pred_svm)

# Interactive ROC curve plot
print("Generating interactive ROC curve...")
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
auc_rf = auc(fpr_rf, tpr_rf)
auc_svm = auc(fpr_svm, tpr_svm)

fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines', name=f"Random Forest (AUC = {auc_rf:.2f})"))
fig.add_trace(go.Scatter(x=fpr_svm, y=tpr_svm, mode='lines', name=f"SVM (AUC = {auc_svm:.2f})"))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Random Chance", line=dict(dash='dash')))

fig.update_layout(
    title="Interactive ROC Curve",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    width=800,
    height=600,
    template="plotly_white"
)

fig.write_html(os.path.join(output_dir, "interactive_roc_curve.html"))
print("ROC curve saved as interactive HTML in output directory.")
