# Bike Sharing Rental Demand Prediction

**Author:** Srishakti Awanti

---

## Project Overview

This project predicts total bike rental demand using historical trip data, weather conditions, calendar variables, and engineered time-based features. The workflow covers data inspection, cleaning, exploratory data analysis, feature engineering, model training, hyperparameter tuning, evaluation, and Streamlit-based deployment.

The project is designed to help understand what drives bike rental demand and to support better fleet planning and resource allocation.

---

## Problem Statement

Bike-sharing systems experience demand that varies by hour, season, weather, and day type. This project builds a machine learning regression pipeline to predict the total number of rentals (`cnt`) from the available contextual features.

---

## Files Included in This Project

### Main scripts
- `clean_and_analyze.py` — initial data cleaning and analysis
- `eda_analysis_v2.py` — exploratory data analysis and visual reporting
- `feature_engineering_v2.py` — feature engineering and preprocessing
- `hyperparameter_tuning.py` — model tuning with RandomizedSearchCV
- `model_building.py` — model training and comparison
- `inspect_dates.py` — date and temporal validation checks
- `check_env.py` — environment verification
- `check_types.py` — data type inspection
- `summarize_results.py` — result summary generation
- `app.py` — Streamlit application entry point
- `streamlit_app.py` — alternate Streamlit interface

### Notebook
- `Bike_Sharing_Full_Process.ipynb` — end-to-end project notebook

### Data files
- `Dataset.csv` — raw dataset
- `cleaned_dataset.csv` — cleaned intermediate dataset
- `cleaned_bike_data.csv` — cleaned bike-sharing dataset
- `processed_bike_data.csv` — final processed training dataset

### Folders
- `images/` — plots, charts, and analysis outputs
- `models/` — saved trained models

### Other project files
- `requirements.py` — dependency list / environment support file
- `README.md` — project documentation

---

## Dataset Description

The dataset includes hourly bike rental records and related predictors such as:

- `yr`
- `mnth`
- `hr`
- `holiday`
- `weekday`
- `workingday`
- `season`
- `weathersit`
- `temp`
- `atemp`
- `hum`
- `windspeed`
- `casual`
- `registered`
- `cnt`

Target variable:
- `cnt` — total count of bike rentals

---

## Project Workflow

### 1. Data Inspection and Cleaning
The raw dataset was checked for:
- missing values
- invalid values
- incorrect data types
- inconsistent date formats

Cleaning steps included:
- type conversion
- missing value handling
- categorical normalization
- date parsing
- duplicate and quality checks

### 2. Exploratory Data Analysis
The data was analyzed using:
- correlation heatmaps
- rental distribution plots
- hourly demand trend plots
- actual vs predicted plots
- feature importance plots

### 3. Feature Engineering
The following transformations were applied:
- one-hot encoding for categorical variables
- cyclic encoding for time-based variables
- scaling for weather-related numeric features
- removal of non-essential columns

### 4. Model Building
Regression models were trained and compared, including:
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

### 5. Hyperparameter Tuning
Random Forest was optimized using `RandomizedSearchCV` to improve prediction quality and generalization.

### 6. Evaluation
Model performance was assessed using:
- R² Score
- RMSE
- MAE

### 7. Deployment
The final model was integrated into a Streamlit app for interactive rental demand prediction.

---

## Model Performance Summary

The Random Forest model performed best overall and was selected as the final model for deployment.

Key evaluation metrics were compared across models using:
- `model_comparison_r2.png`
- `model_comparison_rmse.png`

---

## How to Run the Project

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run preprocessing and modeling scripts
```bash
python clean_and_analyze.py
python eda_analysis_v2.py
python feature_engineering_v2.py
python hyperparameter_tuning.py
python model_building.py
```

### Run the Streamlit app
```bash
streamlit run app.py
```

Or, if you are using the alternate interface:
```bash
streamlit run streamlit_app.py
```

---

## Output Artifacts

Typical generated outputs include:
- cleaned datasets
- processed datasets
- trained model files in `models/`
- analysis charts in `images/`

---

## Project Structure

```text
Bike_Sharing_Project/
├── images/
├── models/
├── app.py
├── Bike_Sharing_Full_Process.ipynb
├── check_env.py
├── check_types.py
├── clean_and_analyze.py
├── cleaned_bike_data.csv
├── cleaned_dataset.csv
├── Dataset.csv
├── eda_analysis_v2.py
├── feature_engineering_v2.py
├── hyperparameter_tuning.py
├── inspect_dates.py
├── model_building.py
├── processed_bike_data.csv
├── README.md
├── requirements.py
├── streamlit_app.py
└── summarize_results.py
```

---

## Conclusion

This project provides a complete machine learning pipeline for predicting bike rental demand. It combines data cleaning, feature engineering, model tuning, and deployment into one structured workflow.

**Author:** Srishakti Awanti
