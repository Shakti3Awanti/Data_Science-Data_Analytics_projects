import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load
try:
    df = pd.read_csv('processed_bike_data.csv')
    print("Data loaded.")
except FileNotFoundError:
    print("Run preprocessing first.")
    exit()

target = 'cnt'
drop_cols = ['casual', 'registered', 'cnt']
X = df.drop(columns=drop_cols, errors='ignore')
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid for Random Search
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestRegressor(random_state=42)

print("Starting Randomized Search...")
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
                               n_iter=20, cv=3, verbose=2, random_state=42, n_jobs=-1)

rf_random.fit(X_train, y_train)

print("Best Parameters:", rf_random.best_params_)

best_rf = rf_random.best_estimator_

# Evaluation
y_pred = best_rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Optimized Model Performance ---")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# Save best model
joblib.dump(best_rf, 'models/best_random_forest.joblib')
print("Saved best_random_forest.joblib")

# Feature Importance
importances = best_rf.feature_importances_
feature_names = X.columns
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

print("\nTop 5 Features:")
print(feature_df.head())

plt.figure(figsize=(10, 8))
# Plot top 15
import seaborn as sns
sns.barplot(x='Importance', y='Feature', data=feature_df.head(15))
plt.title('Feature Importance (Optimized RF)')
plt.tight_layout()
plt.savefig('images/feature_importance.png')
print("Saved feature_importance.png")
