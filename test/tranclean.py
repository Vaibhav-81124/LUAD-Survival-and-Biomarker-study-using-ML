import os
import pandas as pd

data_dir = "../output"
clinical_raw = pd.read_csv(os.path.join(data_dir, "clinical_cleaned.csv"), header=None)
# Read the raw clinical file (assuming CSV here)

# 1. Transpose first
clinical_df = clinical_raw.transpose()

# 2. Set the first row as column names
clinical_df.columns = clinical_df.iloc[0]

# 3. Drop the first row and reset index
clinical_df = clinical_df.drop(clinical_df.index[0]).reset_index().rename(columns={"index": "attrib_name"})

# Now you can use clinical_df normally
print("✅ Cleaned clinical_df columns:", clinical_df.columns.tolist())

