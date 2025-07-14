import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 1. Loading the dataset
df = pd.read_csv("/content/SET-11.csv")
print("Original Dataset:")
print(df)
print(f"Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")

# 2. Replace missing values in 'Math' and 'Science' with their column means
df['Math'].fillna(df['Math'].mean(), inplace=True)
df['Science'].fillna(df['Science'].mean(), inplace=True)
print(f"\nAfter handling missing values:")
print(df)

# 3. Remove duplicate entries
df_before_dedup = df.copy()
df.drop_duplicates(inplace=True)
print(f"\nAfter removing duplicates:")
print(f"Rows before: {len(df_before_dedup)}, Rows after: {len(df)}")
print(df)

# 4. Discretize Science marks BEFORE normalization (using original values)
def categorize_science(score):
    if score < 50:
        return 'Poor'
    elif score < 70:
        return 'Average'
    elif score < 90:
        return 'Good'
    else:
        return 'Excellent'

df['Science_Category'] = df['Science'].apply(categorize_science)
print(f"\nScience categories:")
print(df['Science_Category'].value_counts())

# 5. Normalize numerical columns using Min-Max normalization
num_cols = df.select_dtypes(include=[np.number]).columns
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print(f"\nAfter normalization:")
print(df)

# 6. Smooth noisy data using binning (equal-width binning on Math)
# Create bins and smooth by replacing with bin means
def smooth_column_binning(series, bins=3):
    """Smooth data using equal-width binning with bin means"""
    # Create bins
    bin_edges = np.linspace(series.min(), series.max(), bins + 1)
    bin_labels = [f'Bin_{i+1}' for i in range(bins)]
    
    # Assign to bins
    binned = pd.cut(series, bins=bin_edges, labels=bin_labels, include_lowest=True)
    
    # Calculate bin means
    bin_means = series.groupby(binned).mean()
    
    # Replace values with bin means
    smoothed = series.copy()
    for bin_label in bin_labels:
        mask = binned == bin_label
        smoothed[mask] = bin_means[bin_label]
    
    return smoothed, binned, bin_means

# Apply smoothing to Math column
df['Math_Smoothed'], df['Math_Bin'], math_bin_means = smooth_column_binning(df['Math'])

print(f"\nBinning results for Math:")
print(f"Bin means: {math_bin_means}")
print(f"Smoothed data:")
print(df[['Name', 'Math', 'Math_Bin', 'Math_Smoothed']])

# Final dataset
print(f"\nFinal processed dataset:")
print(df)



