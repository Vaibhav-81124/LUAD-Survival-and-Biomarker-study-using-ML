# LUAD-Survival-and-Biomarker-study-using-ML
This project explores the integration of RNA-seq, mutation, and clinical data to predict survival outcomes in Lung Adenocarcinoma (LUAD) patients using machine learning techniques.

**Objectives**
Integrate multi-omics datasets for LUAD.
Train and evaluate machine learning models (Random Forest, SVM, Voting Classifier).
Identify and rank genes relevant to survival prediction.
Perform pathway and functional enrichment analysis on key genes.
Visualize patient-wise gene presence and stratify survival trends.
Deploy an interactive Streamlit dashboard for result exploration.

**Methodology**
Data Sources: RNA-seq, mutation, and clinical datasets from TCGA.
Preprocessing: Transposed and cleaned gene data; merged with clinical outcomes.
Modeling: Used Random Forest, SVM, and Voting Classifier with accuracy ~62–66%.
Feature Extraction: Top genes ranked using feature importance from RF.
Visualization:
Confusion matrices, ROC curves, boxplots, and clustered heatmaps.
Kaplan-Meier survival plots stratified by gene expression.
Functional Analysis: Performed using g:Profiler for biological pathway interpretation.

**Key Results**
Identified genes (e.g., BMP5, ZNF555, GGTLC1, CTLA4) are strongly associated with survival or death outcomes.
Kaplan-Meier plots revealed significant gene-wise differences in survival trends.
Enrichment analysis suggested links to apoptosis, immune response, and transcription regulation.

**Tools and Libraries**
Python, pandas, numpy, scikit-learn, matplotlib, seaborn, lifelines, Streamlit, xgboost, plotly, g:Profiler for functional enrichment.
