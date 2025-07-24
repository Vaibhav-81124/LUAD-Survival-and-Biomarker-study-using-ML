import pandas as pd
import os

# Set paths
data_dir = "../data"
output_dir = "../output"
driver_file = os.path.join(data_dir, "driverdata.csv")  # Replace with actual filename

# Load driver gene table
print("📥 Loading driver gene data...")
driver_df = pd.read_csv(driver_file)

# Make sure gene column exists and is uppercase for consistency
driver_df['gene'] = driver_df['gene'].str.upper()

# Filter to keep strong evidence genes
filtered_drivers = driver_df[
    (driver_df["CGC"] == 1) | 
    (driver_df["NCG6.0"] == 1) | 
    (driver_df["multiomics"] == 1)
]

# Save filtered driver gene list
driver_genes = filtered_drivers['gene'].drop_duplicates().reset_index(drop=True)
driver_genes_df = pd.DataFrame(driver_genes, columns=["gene"])
driver_genes_df.to_csv(os.path.join(output_dir, "luad_driver_genes.csv"), index=False)

print("✅ Driver genes extracted and saved to /output/luad_driver_genes.csv")
