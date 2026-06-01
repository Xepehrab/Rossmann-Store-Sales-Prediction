# Rossmann Store Sales Prediction Pipeline

This repository contains a machine learning pipeline designed to predict 6 weeks of daily sales for 1,115 Rossmann drug stores across Germany. The project is based on the famous Kaggle time-series forecasting competition and focuses heavily on temporal feature engineering and strict chronological validation to prevent data leakage.

## 📂 Project Structure

- `data/`
  - `train.csv` : Historical daily sales data.
  - `store.csv` : Static information about the stores (StoreType, Assortment, etc.).
- `prepare.py` : Data cleaning and feature engineering module. Extracts time-series data and handles missing values.
- `train_xgb.py` : The master controller script. Handles the validation split, One-Hot Encoding, model training, and evaluation.
- `README.md` : Project documentation.

## 🛠️ Methodology & Technical Highlights

Because retail sales are heavily dependent on time and seasonality, this project treats the challenge as a tabular regression problem enriched with continuous time-derived features.

* **Duration Feature Engineering:** Calculates continuous "status effects" rather than static timestamps. For example, it calculates the exact number of months a competitor has been open on any given transaction day.
* **Strict Time-Series Validation:** Implements a custom chronological split that holds out the final 42 days of data. This perfectly simulates the Kaggle test environment and prevents future-data leakage (predicting the past using future knowledge).
* **Target Transformation:** Trains the model on the logarithmic transformation of sales (`np.log1p`) to naturally optimize for the competition's percentage-based evaluation metric.
* **Robust Categorical Encoding:** Utilizes an integrated `scikit-learn` One-Hot Encoding pipeline that strictly fits to the training data and safely transforms the validation set, ensuring no dimension-mismatch errors if new categories appear.
* **Lightweight Architecture:** Uses a `RandomForestRegressor` and native standard libraries, keeping the environment lightweight and easy to reproduce without complex external dependencies.

## ⚙️ Installation & Requirements

This project is designed to run locally using native Python. No complex external dependencies or `requirements.txt` installations are required beyond standard Data Science libraries:
* `pandas`
* `numpy`
* `scikit-learn`

Ensure your Python environment is active and these standard libraries are installed.

## 🚀 Usage

The pipeline is modular but executed through a single master controller script. The data preparation functions are imported directly into the training script, meaning you do not need to generate intermediate CSV files.

To prepare the data, perform the time-series split, train the Random Forest model, and evaluate the final score, run the following command in your terminal:

bash
python train_xgb.py


## 📊 Evaluation Metric: RMSPE

The model is evaluated using Kaggle's custom **Root Mean Square Percentage Error (RMSPE)**. 

Unlike standard RMSE, this metric penalizes *percentage* errors rather than absolute dollar amounts. This ensures that an error of €100 on a €500 sales day is weighted exactly the same as a €1,000 error on a €5,000 sales day. 

The final RMSPE score is mathematically reversed from log-space and printed directly to the console at the end of the script execution.