import pandas as pd

try:
    df = pd.read_csv('Dataset.csv', encoding='latin1')
    print("Dataset loaded.")
except:
    print("Failed to load.")
    exit()

# Check dteday
print("\n--- dteday inspection ---")
# Sample 20 values from start, middle, end
print("First 10:", df['dteday'].head(10).tolist())
print("Last 10:", df['dteday'].tail(10).tolist())
# Check random sample
print("Random 10:", df['dteday'].sample(10).tolist())

# Check unique lengths
lengths = df['dteday'].astype(str).str.len().unique()
print(f"Unique lengths of dteday strings: {lengths}")

# Check value content for those with length 10 vs others?
for l in lengths:
    sample = df[df['dteday'].astype(str).str.len() == l]['dteday'].head(5).tolist()
    print(f"Length {l} samples: {sample}")
