import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)
import plotly.graph_objects as go

# Set paths
data_dir = "../output"
output_dir = "../output"

# Load labeled data
print("Loading labeled dataset...")
data = pd.read_csv(os.path.join(data_dir, "merged_labeled_data.csv"))

# Prepare features and labels
X = data.drop(columns=["attrib_name", "status"])
y = data["status"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train individual models
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Training SVM...")
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Voting classifier
print("Training Voting Classifier...")
voting_model = VotingClassifier(
    estimators=[('rf', rf_model), ('svm', svm_model)],
    voting='soft'
)
voting_model.fit(X_train, y_train)
y_pred_vote = voting_model.predict(X_test)
y_score_vote = voting_model.predict_proba(X_test)[:, 1]

# Evaluation
acc = accuracy_score(y_test, y_pred_vote)
report = classification_report(y_test, y_pred_vote)
cm = confusion_matrix(y_test, y_pred_vote)

print(f"\nVoting Classifier Accuracy: {acc:.4f}")
print("Classification Report:")
print(report)

# Save confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Voting Classifier Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "voting_confusion_matrix.png"))
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_score_vote)
roc_auc = auc(fpr, tpr)

fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                         name=f"Voting Classifier (AUC = {roc_auc:.2f})"))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                         name="Random Chance", line=dict(dash='dash')))

fig.update_layout(
    title="Voting Classifier ROC Curve",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    width=800,
    height=600,
    template="plotly_white"
)
fig.write_html(os.path.join(output_dir, "voting_roc_curve.html"))

# Save evaluation report
with open(os.path.join(output_dir, "voting_classifier_report.txt"), "w") as f:
    f.write(f"Voting Classifier Accuracy: {acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("Hybrid Voting model results saved to output directory ✅")

