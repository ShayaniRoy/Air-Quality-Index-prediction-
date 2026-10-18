import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier

# --- 1. Load Data ---
data_path = 'mainaqiexmaindataset.csv' if os.path.exists('mainaqiexmaindataset.csv') else 'mainaqimaindataset.csv'
df = pd.read_csv(data_path)

print("Original Dataset Shape:", df.shape)

# --- 2. Data Preprocessing & Mapping ---
# Map categorical target and season columns to numeric values
if 'AQI_Bucket' in df.columns:
    df['AQI_Bucket'] = df['AQI_Bucket'].map({
        "Moderate": 1, 
        "Unhealthy": 2, 
        "Unhealthy for strong people": 3, 
        "Hazardous": 4
    })

if 'season' in df.columns:
    df['season'] = df['season'].map({
        'Winter': 1, 
        'Summer': 2, 
        'Monsoon': 3, 
        'Post-Monsoon': 4
    })

# Drop unused non-numeric columns
cols_to_drop = ['date_time', 'date', 'time']
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# Drop any remaining rows with missing values
df = df.dropna().reset_index(drop=True)

# --- 3. Feature & Target Split ---
X = df.drop(columns=['AQI_Bucket'])
y = df['AQI_Bucket']

# Label encode target just to ensure continuous 0-indexed classes if needed
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# --- 4. Model Training & Evaluation ---
print("\n--- Training Models ---")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train.values, y_train)
y_lr = lr_model.predict(X_test.values)
lr_acc = accuracy_score(y_test, y_lr)
print(f"Accuracy score for Logistic Regression = {lr_acc * 100.0:.2f}%")

# AdaBoost Classifier
ab_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ab_model.fit(X_train.values, y_train)
y_ab = ab_model.predict(X_test.values)
ab_acc = accuracy_score(y_test, y_ab)
print(f"Accuracy score for Ada Boost = {ab_acc * 100.0:.2f}%")

# Voting Classifier (Ensemble)
stack_modl = VotingClassifier(
    estimators=[
        ('RR', LogisticRegression(max_iter=1000)),
        ('ab', AdaBoostClassifier(n_estimators=100, random_state=42))
    ],
    voting='soft'
)
stack_modl.fit(X_train.values, y_train)
y_stack_modl = stack_modl.predict(X_test.values)
stack_acc = accuracy_score(y_test, y_stack_modl)
print(f"Accuracy score for Voting Ensemble = {stack_acc * 100.0:.2f}%\n")

# --- 5. Interactive Sample Prediction ---
print("--- Sample Prediction on Test Set ---")
try:
    n = int(input("Enter an index number from test set for prediction (e.g., 0 to {}): ".format(len(X_test)-1)))
    if 0 <= n < len(X_test):
        sample_input = X_test.iloc[[n]]
        prediction = stack_modl.predict(sample_input.values)
        
        # Map back to readable label
        bucket_mapping = {0: "Moderate", 1: "Unhealthy", 2: "Unhealthy for strong people", 3: "Hazardous"}
        pred_label = bucket_mapping.get(prediction[0], "Unknown")
        print(f"Predicted AQI Bucket: {pred_label}")
    else:
        print("Invalid index entered.")
except ValueError:
    print("Please enter a valid integer.")