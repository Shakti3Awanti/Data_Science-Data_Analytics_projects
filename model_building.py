import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Create models directory
if not os.path.exists('models'):
    os.makedirs('models')

# Load data
print("Loading data...")
try:
    df = pd.read_csv('processed_bike_data.csv')
except FileNotFoundError:
    print("Error: processed_bike_data.csv not found.")
    exit()

# Define Target and Features
# Target is 'cnt'
# We must DROP 'casual' and 'registered' because cnt = casual + registered (Data Leakage)
target = 'cnt'
drop_cols = ['casual', 'registered', 'cnt'] 
# Note: 'cnt' is in drop_cols just to define X, but we keep it for y.

X = df.drop(columns=drop_cols, errors='ignore')
y = df[target]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Initialize Models
models = {
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

print("\n--- Model Training & Evaluation ---")
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    
    print(f"{name} Results:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")
    
    # Save model
    filename = f"models/{name.replace(' ', '_').lower()}.joblib"
    joblib.dump(model, filename)
    print(f"  Saved model to {filename}")

# Comparison Visualization
print("\n--- Generating Comparison Plot ---")
results_df = pd.DataFrame(results).T
print(results_df)

plt.figure(figsize=(10, 6))
results_df['R2'].plot(kind='barh', color='skyblue')
plt.title('Model Comparison - R2 Score')
plt.xlabel('R2 Score')
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('images/model_comparison_r2.png')

plt.figure(figsize=(10, 6))
results_df['RMSE'].plot(kind='barh', color='salmon')
plt.title('Model Comparison - RMSE (Lower is Better)')
plt.xlabel('RMSE')
plt.tight_layout()
plt.savefig('images/model_comparison_rmse.png')

print("\nModel building complete. Results saved.")
