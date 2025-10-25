# Auto MPG Regression Analysis

## Project Overview

This project aims to predict the fuel efficiency (`mpg`) of automobiles based on various features such as engine size, horsepower, weight, acceleration, model year, and origin. Multiple regression models were applied to determine the best-performing algorithm.

Dataset link: [Car Mileage Prediction Dataset](https://www.kaggle.com/datasets/uciml/autompg-dataset)


The dataset used is `auto-mpg.csv` containing 398 car entries with 9 columns:

* **mpg**: Miles per gallon (target variable)
* **cylinders**: Number of cylinders in the engine
* **displacement**: Engine displacement (cubic inches)
* **horsepower**: Engine horsepower
* **weight**: Vehicle weight (lbs)
* **acceleration**: Time to accelerate from 0 to 60 mph
* **model year**: Year of car model
* **origin**: Origin of the car (1 = USA, 2 = Europe, 3 = Japan)
* **car name**: Name of the car (dropped for modeling)

---

## Steps Performed

### 1. Data Loading & Exploration

* Loaded the dataset using `pandas`.
* Checked for null values and duplicates: no missing values in most columns except `'horsepower'` had `'?'` entries.
* Converted `'horsepower'` to numeric and replaced missing values with the median.

### 2. Data Cleaning

* Removed the `'car name'` column as it is non-numeric and not directly useful for prediction.
* Converted all numeric features to appropriate types.
* Identified skewness in features using `scipy.stats.skew`.
* Detected and capped outliers in all numeric columns using the IQR method.

### 3. Feature Transformation

* Applied **Power Transformer (Yeo-Johnson)** to reduce skewness in `'horsepower'` and `'origin'`.
* Scaled features for models requiring normalization (SVR, KNN) using `StandardScaler`.

### 4. Model Preparation

* Split the data into train and test sets (80/20 split).
* Features (`X`) = all columns except `'mpg'`
* Target (`y`) = `'mpg'`

### 5. Model Training & Evaluation

Multiple regression models were trained and evaluated using **R²** and **RMSE** metrics:

| Model                       | R²     | RMSE   |
| --------------------------- | ------ | ------ |
| Linear Regression           | 0.8703 | 2.6407 |
| Decision Tree Regressor     | 0.7932 | 3.3344 |
| Random Forest Regressor     | 0.9124 | 2.1706 |
| Gradient Boosting Regressor | 0.8773 | 2.5689 |
| XGBoost Regressor           | 0.8704 | 2.6241 |
| K-Nearest Neighbors         | 0.9020 | 2.2957 |
| Support Vector Regressor    | 0.9244 | 2.0155 |
| Ridge Regression            | 0.8703 | 2.6411 |
| Lasso Regression            | 0.8703 | 2.6411 |

**Observations:**

* **SVR** gave the best performance with highest R² and lowest RMSE.
* Ensemble methods like **Random Forest** and **Gradient Boosting** performed very well.
* Simple linear models (Linear, Ridge, Lasso) performed decently but not as well as non-linear methods.

---

## Libraries Used

* `pandas`, `numpy` – Data manipulation
* `matplotlib`, `seaborn` – Data visualization
* `scipy.stats` – Skewness calculation
* `sklearn` – Preprocessing, model building, and evaluation
* `xgboost` – XGBoost Regressor

---

## Conclusion

* This project demonstrates how data cleaning, handling missing values, outlier treatment, skewness correction, and model selection affect predictive performance.
* Non-linear regression models like **SVR** and ensemble methods like **Random Forest** perform better than linear models for predicting car fuel efficiency.

---
