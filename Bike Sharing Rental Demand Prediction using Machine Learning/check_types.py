import pandas as pd

df = pd.read_csv('cleaned_bike_data.csv')
print("--- Data Types ---")
print(df.dtypes)
print("\n--- Unique Values in 'mnth' ---")
print(df['mnth'].unique())
print("\n--- Unique Values in 'yr' ---")
print(df['yr'].unique())
print("\n--- Unique Values in 'holiday' ---")
print(df['holiday'].unique())
print("\n--- Unique Values in 'workingday' ---")
print(df['workingday'].unique())
