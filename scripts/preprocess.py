import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Define file paths (relative to project root)
mutation_path = r"C:\Users\ME\Desktop\min_proj\Datasets\mutation.cbt"
rnaseq_path = r"C:\Users\ME\Desktop\min_proj\Datasets\RNA.cct"
output_path = "../outputs/processed_luad.csv"

# Step 1: Load the mutation data
print(" Loading mutation data...")
mutation_df = pd.read_csv(mutation_path, sep='\t')
print(f" Mutation data shape: {mutation_df.shape}")

# Step 2: Load the RNA-seq data
print("📥 Loading RNA-seq data...")
rnaseq_df = pd.read_csv(rnaseq_path, sep='\t')
print(f" RNA-seq data shape: {rnaseq_df.shape}")

# Step 3: Merge on gene name
print(" Merging on 'gene' column...")
merged_df = pd.merge(mutation_df, rnaseq_df, on='gene', how='inner')
print(f" Merged dataset shape: {merged_df.shape}")

# Step 4: Fill any missing values
merged_df.fillna(0, inplace=True)

# Step 5: Normalize feature columns
print(" Normalizing features...")
gene_col = merged_df['gene']
features = merged_df.drop('gene', axis=1)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 6: Reassemble final dataframe
processed_df = pd.DataFrame(features_scaled, columns=features.columns)
processed_df.insert(0, 'gene', gene_col)

# Step 7: Save to outputs
print(" Saving preprocessed data to CSV...")
processed_df.to_csv(output_path, index=False)
print(f" Done! File saved at: {output_path}")

