import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             roc_curve, auc)
import plotly.graph_objects as go

# Set paths
data_dir = "../output"
output_dir = "../output"

# Load data
print("Loading labeled dataset...")
data = pd.read_csv(os.path.join(data_dir, "merged_labeled_data.csv"))

# Drop non-feature columns and extract labels
X_full = data.drop(columns=["attrib_name", "status"])
y = data["status"]

# Feature Selection using Random Forest
print("Selecting top features with Random Forest...")
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X_full, y)

selector = SelectFromModel(rf_selector, prefit=True, max_features=50, threshold=-np.inf)
X_selected = selector.transform(X_full)
selected_features = X_full.columns[selector.get_support()]
X = pd.DataFrame(X_selected, columns=selected_features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(probability=True, random_state=42)
voting_model = VotingClassifier(
    estimators=[('rf', rf_model), ('svm', svm_model)],
    voting='soft'
)

models = {
    "RandomForest": rf_model,
    "SVM": svm_model,
    "VotingClassifier": voting_model
}

# Evaluation report text
report_text = ""

# ROC setup
fig = go.Figure()

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    # Accuracy and report
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    report_text += f"\n{name} Accuracy: {acc:.4f}\n"
    report_text += f"{report}\n"

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name.lower()}_confusion_matrix.png"))
    plt.close()

    # Add ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC = {roc_auc:.2f})"))

# Random baseline line
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Random Chance", line=dict(dash='dash')))

fig.update_layout(
    title="Model Comparison - ROC Curves",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    width=800,
    height=600,
    template="plotly_white"
)

# Save files
fig.write_html(os.path.join(output_dir, "comparison_roc_curves.html"))
with open(os.path.join(output_dir, "model_comparison_report.txt"), "w") as f:
    f.write(report_text)

print("Model comparison plots and report saved to output directory ✅")

# Feature importance plot (only for RF)
print("Plotting feature importances...")
importances = rf_model.feature_importances_
top_indices = np.argsort(importances)[-10:]
top_features = X.columns[top_indices]
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
    sns.boxplot(x=y, y=X[feature])
    plt.title(f"Expression of {feature} by Survival Status")
    plt.xlabel("Survival Status (0=Deceased, 1=Alive)")
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
