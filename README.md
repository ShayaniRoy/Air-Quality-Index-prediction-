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
Key Highlights & Performance MetricsAQI Risk Classification Accuracy: >99.9% across Random Forest, AdaBoost, and Soft/Hard Voting Ensembles.Continuous Regression: Linear Regression baseline ($R^2 / \text{Accuracy} = 73.12\%$) paired with Support Vector Regression (SVR) for bounded continuous continuous target modeling.Feature Engineering: Automated generation of multi-pollutant $lag_1$, $lag_7$, and 7-day rolling moving averages ($ma_7$) capturing autocorrelation and atmospheric persistence.Decomposition & Forecasting: Structural trend and multi-seasonal (daily, weekly, yearly) forecasting using Facebook Prophet across 680k+ historical observations.Repository Structure.
├── data/
│   ├── raw/                  # Original unprocessed data
│   └── processed/            # Engineery features (lags, moving averages)
├── notebooks/
│   ├── 01_eda_and_decomposition.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classification_ensembles.ipynb
│   └── 04_prophet_forecasting.ipynb
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py      # Data loading, cleaning, and lag generation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifiers.py    # Voting Ensembles, AdaBoost, Random Forest
│   │   ├── regressors.py     # Linear, Ridge, and SVR implementations
│   │   └── forecaster.py     # FB Prophet wrapper and decomposition logic
│   └── utils/
│       ├── metrics.py        # Custom evaluation routines
│       └── visualization.py  # Trend, pie, and seasonal plotting tools
├── reports/
│   └── figures/              # Saved output plots (confusion matrices, trends)
├── requirements.txt
├── LICENSE
└── README.md
Dataset & Feature Engineering ArchitectureThe model ingests multi-pollutant environmental sensor datasets containing continuous pollutant concentrations and temporal attributes:Schema Overviewdate, pm25, pm10, no2, so2, co, o3, hour, day, month, year, weekday, AQI, season, pm25_lag1, pm25_lag7, pm10_lag1, pm10_lag7, no2_lag1, no2_lag7, so2_lag1, so2_lag7, co_lag1, co_lag7, o3_lag1, o3_lag7, pm25_ma7, pm10_ma7, no2_ma7, so2_ma7, co_ma7, o3_ma7, AQI_BucketEngineering PipelineAutoregressive Lags: $1$-step and $7$-step historical lags computed per pollutant ($\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3$) to embed temporal dependencies.Moving Averages: $7$-day rolling averages ($\text{ma}_7$) calculated across sensor feeds to suppress high-frequency noise while preserving macro-level air quality shifts.Categorical Risk Labeling: Implements standardized EPA/CPCB AQI buckets:Moderate: ~44%Unhealthy: ~33%Unhealthy for strong people: ~23%Hazardous: <1%Machine Learning & Forecasting Architecture[Raw Pollutant Data] ──► [Feature Pipeline (Lags/MA7)] ┬──► [Classification Ensembles] ──► [AQI Risk Bucket]
                                                         ├──► [SVR & Linear Regression]  ──► [Continuous AQI]
                                                         └──► [FB Prophet Engine]       ──► [Multi-Season Forecast]
1. Classification EnsemblesEvaluates health risk buckets using weighted soft-voting ensembles combining Logistic Regression, Decision Trees, Bagging, Random Forest, and AdaBoost.Outcome: Achieved $>99.9\%$ validation accuracy due to strong feature signals provided by short-term lag vectors and rolling moving averages.2. Continuous AQI ModelingEmploys SVR alongside Ridge/Linear Regression to establish continuous AQI baseline predictions, utilizing temporal backtesting with rolling validation windows to safeguard against data leakage.3. Prophet Time-Series ForecastingDeconstructs multi-year signals (2015–2023) into decoupled seasonal vectors:
$$\text{AQI}(t) = g(t) + s_{daily}(t) + s_{weekly}(t) + s_{yearly}(t) + \epsilon_t$$
Provides long-term hourly and daily forecasting with uncertainty intervals.Visualizations & InsightsClass Distribution: Exploratory analysis highlights class imbalances, predominant occurrence in Moderate (44%) and Unhealthy (33%) tiers, and extreme sparse events in Hazardous (<1%).Historical Yearly Trends: Macro analysis spanning 2015–2023 reveals distinct winter-peak seasonal cyclicality.Additive Component Decomposition: Prophet decomposition isolating macro trends, day-of-week traffic impacts, and annual seasonal spikes.Quick Start1. Prerequisites & Installation# Clone the repository
git clone https://github.com/your-username/aqi-forecasting-pipeline.git
cd aqi-forecasting-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
2. Running the PipelineTo run data preparation, feature engineering, and model training:python -m src.data_pipeline
python -m src.models.classifiers
python -m src.models.forecaster
LicenseDistributed under the MIT License. See LICENSE for more information.
