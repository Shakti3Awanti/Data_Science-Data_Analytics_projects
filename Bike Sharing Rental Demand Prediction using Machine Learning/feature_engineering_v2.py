import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

if not os.path.exists('images'):
    os.makedirs('images')

# Load data
df = pd.read_csv('cleaned_bike_data.csv')

# --- 0. PRE-PROCESSING / FIXING CATEGORICALS ---
print("--- Fixing Categorical Data ---")

# Convert dteday to datetime first to recover info
df['dteday'] = pd.to_datetime(df['dteday'])

# Fix 'mnth' from dteday
# Sometimes 'mnth' has '?', but dteday is valid.
df['mnth'] = df['dteday'].dt.month
print("Fixed 'mnth' using dteday.")

# Fix 'yr'
# Map years to 0 (2011) and 1 (2012)
# If dteday year is 2011 -> 0, 2012 -> 1
df['yr'] = df['dteday'].dt.year.map({2011: 0, 2012: 1})
print("Fixed 'yr' using dteday.")

# Fix 'holiday' ('No', 'Yes', '?')
# Replace '?' with mode (usually 'No')
mode_holiday = df[df['holiday'] != '?']['holiday'].mode()[0]
df['holiday'] = df['holiday'].replace('?', mode_holiday)
# Map to 0/1
df['holiday'] = df['holiday'].map({'No': 0, 'Yes': 1}).astype(int)
print(f"Fixed 'holiday' (imputed '?' with '{mode_holiday}').")

# Fix 'workingday' ('No work', 'Working Day', '?')
# Replace '?' with mode
mode_working = df[df['workingday'] != '?']['workingday'].mode()[0]
df['workingday'] = df['workingday'].replace('?', mode_working)
# Map to 0/1
df['workingday'] = df['workingday'].map({'No work': 0, 'Working Day': 1}).astype(int)
print(f"Fixed 'workingday' (imputed '?' with '{mode_working}').")

# Fix 'weekday' just in case (already numeric but good to ensure consistency)
# 0: Sunday, 1: Monday... 6: Saturday (pandas .dow is 0=Mon, 6=Sun)
# Original dataset: "Weekday Day of the week" (0-6). Let's assume standard starts 0.
# We can just keep the original column if it was int64, which it was.

# --- 1. Feature Engineering ---

# A. Categorical Encoding (Season, Weathersit)
# Season: springer, summer, fall, winter (from unique values or implied)
# Weathersit: 1, 2, 3, 4 (stored as strings or objects in original? check)
# Let's ensure 'weathersit' is clean. 
# It might have '?' too.
if df['weathersit'].dtype == object:
    # Check for '?'
    mode_weather = df[df['weathersit'] != '?']['weathersit'].mode()[0]
    df['weathersit'] = df['weathersit'].replace('?', mode_weather)
    # Check if they are numeric strings '1','2','3','4' or words.
    # Assuming from description 1,2,3,4.
    # If they are words, pd.get_dummies handles them. If numbers, we treat as categorical.
    print("Unique weathersit:", df['weathersit'].unique())

# One-Hot Encoding
df = pd.get_dummies(df, columns=['season', 'weathersit'], prefix=['season', 'weather'], drop_first=True)
print("Applied One-Hot Encoding.")

# B. Cyclic Encoding
def encode_cyclic(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

# Ensure they are numeric
df['hr'] = pd.to_numeric(df['hr'], errors='coerce').fillna(0).astype(int)
df['mnth'] = pd.to_numeric(df['mnth'], errors='coerce').fillna(1).astype(int)
df['weekday'] = pd.to_numeric(df['weekday'], errors='coerce').fillna(0).astype(int)

df = encode_cyclic(df, 'hr', 24)
df = encode_cyclic(df, 'mnth', 12)
df = encode_cyclic(df, 'weekday', 7)
print("Applied Cyclic Encoding.")

# C. Scaling
scale_cols = ['temp', 'atemp', 'hum', 'windspeed']
scaler = MinMaxScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])
print("Scaled features.")

# D. Cleanup
drop_cols = ['dteday', 'instant']
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Save
df.to_csv('processed_bike_data.csv', index=False)
print("Saved processed_bike_data.csv")

# Correlation
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Matrix (Processed)')
plt.tight_layout()
plt.savefig('images/correlation_matrix_processed.png')
print("Saved correlation matrix.")
