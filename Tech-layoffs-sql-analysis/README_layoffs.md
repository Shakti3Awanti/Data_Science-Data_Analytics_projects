# Global Layoffs Data Cleaning and EDA

A SQL-based data cleaning and exploratory data analysis (EDA) project on global layoff data. The workflow standardizes the raw dataset, removes duplicates, handles null and blank values, and then analyzes layoff patterns across companies, industries, countries, stages, and time.

## Project Objective

The goal of this project is to prepare a messy layoffs dataset for analysis and answer business questions such as:

- Which companies had the highest total layoffs?
- Which industries and countries were most affected?
- How did layoffs change over time?
- Which funding stages saw the largest impact?

## Dataset

The dataset used in this project contains the following fields:

- `company`
- `location`
- `industry`
- `total_laid_off`
- `percentage_laid_off`
- `date`
- `stage`
- `country`
- `funds_raised_millions`

## Tools Used

- **SQL (MySQL-compatible)**
- **CSV data source**
- **Exploratory data analysis**
- **Window functions**
- **Data cleaning and transformation**

## Data Cleaning Steps

The SQL script performs the following cleaning tasks:

1. Remove duplicate records using `ROW_NUMBER()`
2. Standardize text fields such as company, industry, and country
3. Convert the `date` field into a proper `DATE` type
4. Handle null and blank values
5. Remove rows where both layoff measures are missing
6. Drop helper columns after cleaning

## Exploratory Analysis Performed

The EDA section includes analysis of:

- Maximum layoffs and full layoff cases
- Total layoffs by company
- Total layoffs by industry
- Total layoffs by country
- Total layoffs by year
- Total layoffs by funding stage
- Rolling monthly layoff totals
- Top companies by layoffs within each year

## Key Insights

This analysis helps reveal:

- Which companies contributed the largest share of layoffs
- Which industries were hit hardest
- Which countries were most affected
- How layoff activity changed across the 2020–2023 period

## File Structure

```text
.
├── layoffs.csv
├── EDA proj.sql
└── README.md
```

## How to Use

1. Import `layoffs.csv` into your MySQL database.
2. Run the SQL script in `EDA proj.sql`.
3. Review the cleaned table and execute the analysis queries.
4. Use the results to build charts, dashboards, or a presentation.

## Notes

- The SQL script uses a staging-table approach for safe transformation.
- The project is suitable for a data analytics portfolio or GitHub job search repository.
- You can extend this project by adding visualizations in Python, Power BI, Tableau, or Looker Studio.

## Author

SriShakti Awanti
