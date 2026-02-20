"""
Rock vs Mine Prediction - Model Training Script
=================================================
Trains a Logistic Regression model on the Sonar dataset to classify
objects as either Rock (R) or Mine (M) based on 60 sonar frequency features.

Based on: Siddhardhan's ML Project Series
Dataset: UCI Sonar Dataset (208 samples, 60 features)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import os

# ─── Configuration ───────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "sonar data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

# ─── 1. Load the Dataset ─────────────────────────────────────────────
print("=" * 60)
print("   Rock vs Mine Prediction - Model Training")
print("=" * 60)

sonar_data = pd.read_csv(DATA_PATH, header=None)

print(f"\n[DATA] Dataset shape: {sonar_data.shape}")
print(f"[DATA] Class distribution:\n{sonar_data[60].value_counts()}")

# ─── 2. Separate Features & Labels ───────────────────────────────────
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# ─── 3. Train-Test Split ─────────────────────────────────────────────
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1
)

print(f"\n[SPLIT] Train set size: {X_train.shape[0]}")
print(f"[SPLIT] Test set size:  {X_test.shape[0]}")

# ─── 4. Feature Scaling ──────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[SCALE] StandardScaler applied to features")

# ─── 5. Train Logistic Regression Model ──────────────────────────────
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, Y_train)

# ─── 6. Evaluate Model ───────────────────────────────────────────────
train_pred = model.predict(X_train_scaled)
train_accuracy = accuracy_score(Y_train, train_pred)

test_pred = model.predict(X_test_scaled)
test_accuracy = accuracy_score(Y_test, test_pred)

print(f"\n[RESULT] Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"[RESULT] Test Accuracy:     {test_accuracy * 100:.2f}%")

# ─── 7. Save Model & Scaler ──────────────────────────────────────────
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print(f"\n[SAVED] Model saved to: {MODEL_PATH}")
print(f"[SAVED] Scaler saved to: {SCALER_PATH}")
print("=" * 60)

# ─── 8. Quick Prediction Test ────────────────────────────────────────
# Sample Rock data (first row from dataset)
rock_sample = sonar_data[sonar_data[60] == "R"].iloc[0, :60].values.reshape(1, -1)
rock_sample_scaled = scaler.transform(rock_sample)
rock_pred = model.predict(rock_sample_scaled)
print(f"\n[TEST] Rock sample prediction: {rock_pred[0]}")

# Sample Mine data (first Mine row from dataset)
mine_sample = sonar_data[sonar_data[60] == "M"].iloc[0, :60].values.reshape(1, -1)
mine_sample_scaled = scaler.transform(mine_sample)
mine_pred = model.predict(mine_sample_scaled)
print(f"[TEST] Mine sample prediction: {mine_pred[0]}")
