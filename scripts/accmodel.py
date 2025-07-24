import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc)
from imblearn.over_sampling import SMOTE
import plotly.graph_objects as go

# Paths
data_dir = "../output"
output_dir = "../test"

# Load data
print("Loading labeled dataset...")
data = pd.read_csv(os.path.join(data_dir, "merged_labeled_data.csv"))

# Features and labels
X = data.drop(columns=["attrib_name", "status"])
y = data["status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE on training data only
print("Applying SMOTE to balance classes...")
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# Define models with class_weight
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
svm_model = SVC(probability=True, class_weight="balanced", random_state=42)

# Voting classifier
voting_model = VotingClassifier(
    estimators=[('rf', rf_model), ('svm', svm_model)],
    voting='soft'
)

models = {
    "RandomForest": rf_model,
    "SVM": svm_model,
    "VotingClassifier": voting_model
}

report_text = ""
fig = go.Figure()

# Train, predict, evaluate
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    report_text += f"\n{name} Accuracy: {acc:.4f}\n"
    report_text += f"{report}\n"

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name.lower()}_improved_cm.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC = {roc_auc:.2f})"))

# Random baseline
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Random Chance", line=dict(dash='dash')))

fig.update_layout(
    title="Improved Model Comparison - ROC Curves",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    width=800,
    height=600,
    template="plotly_white"
)

# Save results
fig.write_html(os.path.join(output_dir, "comparison_roc_curves_improved.html"))
with open(os.path.join(output_dir, "model_comparison_report_improved.txt"), "w") as f:
    f.write(report_text)

print("✅ Improved model comparison files saved!")

