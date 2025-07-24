import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import os

# === Set Directories ===
data_dir = "../output"

# === Load Data ===
st.title("LUAD Survival Prediction Dashboard")
st.markdown("---")

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Overview", "Model Performance", "Feature Insights", "Functional Enrichment", "Survival Analysis", "About"])

# Preload Data
@st.cache_data
def load_data():
    data = pd.read_csv(os.path.join(data_dir, "merged_labeled_data.csv"))
    return data

data = load_data()

# === Section 1: Overview ===
if section == "Overview":
    st.header("Project Overview")
    st.markdown("""
    **Title:** Machine Learning-Based Multi-Omics Integration for Survival Prediction in Lung Adenocarcinoma (LUAD)

    This dashboard presents our mini project where we used Random Forest, SVM, and a Voting Classifier to predict survival outcomes in LUAD patients using RNA-seq and mutation data.

    We also identified important genes, performed functional enrichment, and validated results with Kaplan-Meier survival analysis.
    """)

# === Section 2: Model Performance ===
elif section == "Model Performance":
    st.header("Model Performance")

    for model in ["randomforest", "svm", "votingclassifier"]:
        st.subheader(f"{model.upper()} Confusion Matrix")
        st.image(os.path.join(data_dir, f"{model}_confusion_matrix.png"))

    st.subheader("Interactive ROC Curve")
    st.components.v1.html(open(os.path.join(data_dir, "comparison_roc_curves.html"), 'r').read(), height=600)

    st.subheader("Classification Reports")
    with open(os.path.join(data_dir, "model_comparison_report.txt")) as f:
        st.text(f.read())

# === Section 3: Feature Insights ===
elif section == "Feature Insights":
    st.header("Top Features and Gene Expression")

    st.subheader("Top 10 Features by Random Forest")
    st.image(os.path.join(data_dir, "feature_importance_rf.png"))
    st.subheader("Box Plots of Top Genes")

    top_genes = pd.read_csv(os.path.join(data_dir, "top_important_genes_with_scores.csv"))

    for gene_name in top_genes['Gene'][:10]:  # Correct column access
        image_path = os.path.join(data_dir, f"boxplot_{gene_name}.png")
    if os.path.exists(image_path):
        st.image(image_path)
    else:
        st.warning(f"Image not found for gene: {gene_name}")


    st.subheader("Clustered Heatmap (Top 100 Genes)")
    st.image(os.path.join(data_dir, "clustered_heatmap_top100_scipy.png"))

# === Section 4: Functional Enrichment ===
elif section == "Functional Enrichment":
    st.header("Functional Enrichment Analysis")

    st.image(os.path.join(data_dir, "gprofiler_barplot.png"))
    df = pd.read_csv(os.path.join(data_dir, "gProfiler_hsapiens_11-04-2025_12-30-47__intersections.csv"))
    st.dataframe(df.head(10))

# === Section 5: Survival Analysis ===
elif section == "Survival Analysis":
    st.header("Kaplan-Meier Survival Analysis")

    survival_genes = pd.read_csv(os.path.join(data_dir, "survival_genes_auto.csv"))
    selected_gene = st.selectbox("Select a gene to view its Kaplan-Meier plot:", survival_genes["Gene"].tolist())

    if selected_gene:
        st.image(os.path.join(data_dir, f"{selected_gene}_survival.png"))

# === Section 6: About ===
elif section == "About":
    st.header("About This Project")
    st.markdown("""
    - **Team Size:** 4 members
    - **Duration:** 2 Months (6th Semester Mini Project)
    - **Tools Used:** Python, scikit-learn, pandas, seaborn, lifelines, g:Profiler, Streamlit
    - **Data Source:** TCGA LUAD datasets
    - **Outcome:** Achieved 66% accuracy using Random Forest with gene-level survival insights.
    """)


