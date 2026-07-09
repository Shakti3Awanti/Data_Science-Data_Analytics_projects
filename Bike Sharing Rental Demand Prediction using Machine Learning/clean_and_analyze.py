import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

if not os.path.exists('images'):
    os.makedirs('images')

# Load
df = pd.read_csv('Dataset.csv', encoding='latin1')
print(f"Original shape: {df.shape}")

# 1. Fix Date
# Format appears to be DD-MM-YYYY
df['dteday'] = pd.to_datetime(df['dteday'], dayfirst=True, errors='coerce')
print(f"Date conversion complete. Missing dates: {df['dteday'].isnull().sum()}")

# 2. Fix Numeric Columns
cols_to_fix = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']
for col in cols_to_fix:
    # Coerce to numeric, turning '?' into NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Smart Imputation
# 3a. Casual/Registered
# Check if casual is missing but registered and cnt are present
mask_cas_missing = df['casual'].isnull() & df['registered'].notnull() & df['cnt'].notnull()
df.loc[mask_cas_missing, 'casual'] = df.loc[mask_cas_missing, 'cnt'] - df.loc[mask_cas_missing, 'registered']

# Check if registered is missing but casual and cnt are present
mask_reg_missing = df['registered'].isnull() & df['casual'].notnull() & df['cnt'].notnull()
df.loc[mask_reg_missing, 'registered'] = df.loc[mask_reg_missing, 'cnt'] - df.loc[mask_reg_missing, 'casual']

# If both missing? (Unlikely)
mask_both = df['casual'].isnull() & df['registered'].isnull()
if mask_both.sum() > 0:
    print(f"Rows with both casual and registered missing: {mask_both.sum()}")
    # Drop or impute? Drop for now as it's < 1%
    df = df[~mask_both]

# 3b. Weather cols
# Impute with median
weather_cols = ['temp', 'atemp', 'hum', 'windspeed']
for col in weather_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Imputed {col} with median: {median_val}")

print("\nMissing values after imputation:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 4. Save
df.to_csv('cleaned_bike_data.csv', index=False)
print("Saved cleaned_bike_data.csv")

# 5. Visualizations on CLEAN data
# Correlation
plt.figure(figsize=(12, 10))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix (Cleaned)')
plt.savefig('images/correlation_matrix_final.png')

# Monthly trend
plt.figure(figsize=(12, 6))
# Extract month if needed
df['month_name'] = df['dteday'].dt.month_name()
# Group by month (need sorting)
# simpler: use 'mnth' column if it aligns, but let's use datetime
monthly_cnt = df.groupby(df['dteday'].dt.to_period('M'))['cnt'].sum()
monthly_cnt.plot(kind='line')
plt.title('Total Bike Rentals Over Time')
plt.ylabel('Count')
plt.xlabel('Date')
plt.tight_layout()
plt.savefig('images/time_series_trend.png')

print("EDA Analysis complete. Images saved.")
