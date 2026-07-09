import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

# Create images directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Load the dataset
try:
    df = pd.read_csv('Dataset.csv', encoding='latin1')
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 1. Inspect Data Types and Clean
print("\n--- Initial Info ---")
print(df.info())

# Columns that should be numeric but are object
numeric_candidates = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']

print("\n--- Checking for non-numeric values ---")
for col in numeric_candidates:
    # Force convert to numeric, trace errors
    temp_series = pd.to_numeric(df[col], errors='coerce')
    n_errors = temp_series.isna().sum()
    if n_errors > 0:
        print(f"Column '{col}' has {n_errors} non-numeric entries (will be converted to NaN).")
        # specific examples
        invalid_mask = pd.to_numeric(df[col], errors='coerce').isna()
        print(f"Examples: {df.loc[invalid_mask, col].unique()[:5]}")
    
    # Apply conversion
    df[col] = temp_series

# Convert dteday to datetime
df['dteday'] = pd.to_datetime(df['dteday'], errors='coerce')

# Check for missing values after conversion
print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Fill missing values if any (for now just print, user decided strategy)
# Strategy: user said "Handle missing values: Detect... and apply appropriate imputation"
# If simple errors, maybe drop? Or impute?
# Let's see how many first.

# 2. Outlier Detection (Boxplots)
print("\n--- Generating Outlier Boxplots ---")
plt.figure(figsize=(15, 10))
# Use cleaned numeric cols plus 'cnt'
plot_cols = numeric_candidates + ['cnt']
# ensure they are in df
plot_cols = [c for c in plot_cols if c in df.columns]

for i, col in enumerate(plot_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[col].dropna())
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.savefig('images/outliers_boxplot_cleaned.png')
print("Saved images/outliers_boxplot_cleaned.png")

# 3. Correlation Matrix
print("\n--- Generating Correlation Matrix ---")
plt.figure(figsize=(12, 10))
# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])
if not numeric_df.empty:
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('images/correlation_matrix.png')
    print("Saved images/correlation_matrix.png")
else:
    print("No numeric columns for correlation matrix.")

# Save cleaned data for next steps
df.to_csv('cleaned_dataset.csv', index=False)
print("\nSaved cleaned dataset to 'cleaned_dataset.csv'")
