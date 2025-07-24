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
    report = classification_report(y_true, y_pred, zero_division=0)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(f"Classification Report:\n{report}")

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name.lower()}_confusion_matrix.png"))
    plt.close()

    # Save report
    with open(os.path.join(output_dir, f"{name.lower()}_report.txt"), "w") as f:
        f.write(f"{name} Accuracy: {acc:.4f}\n\n")
        f.write(report)

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

# Feature importance plot
print("Plotting feature importances...")
importances = rf_model.feature_importances_
feature_names = X.columns
top_indices = np.argsort(importances)[-10:]
top_features = feature_names[top_indices]
top_importances = importances[top_indices]

plt.figure(figsize=(8, 6))
sns.barplot(x=top_importances, y=top_features, orient="h")
plt.title("Top 10 Important Features (Random Forest)")
plt.xlabel("Feature Importance")
plt.ylabel("Gene")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance_rf.png"))
plt.close()

# Box plot for top features
print("Generating box plots for top features...")
for feature in top_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=y, y=data[feature])
    plt.title(f"Expression of {feature} by Survival Status")
    plt.xlabel("Survival Status (0=Alive, 1=Deceased)")
    plt.ylabel("Expression Level")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"boxplot_{feature}.png"))
    plt.close()

# Explanation text for box plots
explanation_text = '''
Box Plot for Gene Expression by Survival Status
-----------------------------------------------

Box plots help visualize how gene expression varies between patients who survived vs. those who didn't.

- Useful for identifying potential biomarkers.
- Highlights outliers and expression spread.
- Median line shows central tendency.
- Helps biological interpretation.

This can guide future survival analysis or functional validation.
'''

with open(os.path.join(output_dir, "boxplot_explanation.txt"), "w") as f:
    f.write(explanation_text)

print("All evaluation results and plots saved in output directory.")

