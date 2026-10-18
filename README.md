# 🌬️ Air Quality Index (AQI) Prediction & Time-Series Forecasting Pipeline

An end-to-end machine learning framework designed to analyze, classify, and forecast air pollution metrics. This project implements multiple machine learning paradigms—including multi-class classification, support vector regression, rolling-window backtesting, and advanced time-series forecasting.

---

## 🚀 Key Features & Architecture

The pipeline is structured into four core analytical and predictive modules:

1. **AQI Bucket Classification**: 
   * Categorizes air quality into distinct health risk levels (Moderate, Unhealthy, Hazardous).
   * Evaluates multiple algorithms including **Random Forest, Logistic Regression, Decision Trees, AdaBoost, and Multilayer Perceptrons (MLP)**.
   * Features robust **Voting and Stacking Ensembles** to maximize prediction accuracy.
2. **Continuous Regression (SVR)**: 
   * Predicts continuous AQI values using **Support Vector Regression (SVR)** coupled with feature standardization and error evaluation (RMSE/MAE).
3. **Ridge Regression Backtesting**: 
   * Simulates rolling-window temporal backtesting to evaluate linear model performance dynamically across historical data blocks.
4. **Time-Series Trend Forecasting (Prophet)**: 
   * Leverages **Facebook Prophet** to model seasonality, trends, and generate future hourly/daily AQI forecasts.

---

## 🛠️ Tech Stack

* **Language:** Python 
* **Data Manipulation & Analysis:** Pandas, NumPy, SciPy
* **Machine Learning:** Scikit-Learn (Ensembles, Metrics, Preprocessing)
* **Time-Series Modeling:** Prophet
* **Data Visualization:** Matplotlib, Seaborn, Plotly

---

## 📂 Project Structure

```text
├── main.py                  # Master pipeline script (preprocessing, training, evaluation)
├── mainaqimaindataset.csv   # Primary AQI dataset
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
