import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Load the dataset
df = pd.read_csv("/content/SET-12.csv")
print("Original Dataset:")
print(df)
print(f"\nShape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")

# 2. Fill missing values with column means
print("\n2. Filling missing values with column means:")
# Show what means will be used
numeric_means = df.mean(numeric_only=True)
print(f"Column means: {numeric_means}")

df.fillna(numeric_means, inplace=True)
# Handle non-numeric missing values
df.fillna('Unknown', inplace=True)
print("Missing values filled successfully")

# 3. Drop duplicate records
print("\n3. Removing duplicate records:")
before = df.shape[0]
df.drop_duplicates(inplace=True)
after = df.shape[0]
print(f"Dropped {before - after} duplicate records")
print(f"Dataset shape after deduplication: {df.shape}")

# 4. Apply normalization to numerical columns
print("\n4. Applying normalization:")
num_cols = df.select_dtypes(include=[np.number]).columns
print(f"Numerical columns to normalize: {list(num_cols)}")

scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print("Normalization completed (Min-Max scaling)")

# 5. Discretize scores using binning into 3 categories
print("\n5. Discretizing scores into 3 categories:")
for col in num_cols:
    df[f'{col}_Category'] = pd.cut(df[col], bins=3, labels=['Low', 'Medium', 'High'])
    print(f"{col} categories: {df[f'{col}_Category'].value_counts().to_dict()}")

# 6. Remove noise using binning method (smooth by bin mean)
print("\n6. Smoothing data using binning method:")

def smooth_column_binning(series, bins=3):
    """Smooth data using equal-width binning by replacing values with bin means"""
    bin_edges = np.linspace(series.min(), series.max(), bins + 1)
    bin_labels = [f'Bin{i+1}' for i in range(bins)]
    binned = pd.cut(series, bins=bin_edges, labels=bin_labels, include_lowest=True)
    bin_means = series.groupby(binned).mean()
    smoothed = series.copy()
    for label in bin_labels:
        smoothed[binned == label] = bin_means[label]
    return smoothed, binned

# Apply smoothing to all numeric columns
for col in num_cols:
    df[f'{col}_Smoothed'], df[f'{col}_Bin'] = smooth_column_binning(df[col])
    # Show smoothing effect
    original_std = df[col].std()
    smoothed_std = df[f'{col}_Smoothed'].std()
    reduction = ((original_std - smoothed_std) / original_std) * 100
    print(f"{col}: Noise reduced by {reduction:.1f}% (std: {original_std:.3f} → {smoothed_std:.3f})")

# Display final processed dataset
print("\n" + "="*50)
print("FINAL PROCESSED DATASET")
print("="*50)
print(df)

# Summary of all transformations
print("\n" + "="*50)
print("TRANSFORMATION SUMMARY")
print("="*50)
print(f"✓ Missing values filled with column means")
print(f"✓ {before - after} duplicate records removed")
print(f"✓ {len(num_cols)} numerical columns normalized")
print(f"✓ Categories created for: {', '.join(num_cols)}")
print(f"✓ Data smoothed using binning for: {', '.join(num_cols)}")

# Show key columns for verification
print("\nKey columns in final dataset:")
final_cols = ['Name'] + list(num_cols) + [f'{col}_Category' for col in num_cols] + [f'{col}_Smoothed' for col in num_cols]
print(f"Columns: {final_cols}")
print(f"Total columns: {len(df.columns)}")
