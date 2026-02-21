import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load the dataset
diabetes_data = pd.read_csv('diabetes.csv')

# Note: The dataset contains 8 features and 1 target (Outcome)

# 2. Separate features and target
X = diabetes_data.drop(columns='Outcome', axis=1)
Y = diabetes_data['Outcome']

# 3. Data Standardization
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data

# 4. Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# 5. Train the Models
# Model 1: SVM
clf_svm = svm.SVC(kernel='linear', probability=True)
clf_svm.fit(X_train, Y_train)

# Model 2: Logistic Regression
clf_lr = LogisticRegression()
clf_lr.fit(X_train, Y_train)

# Model 3: Random Forest
clf_rf = RandomForestClassifier(n_estimators=100, random_state=2)
clf_rf.fit(X_train, Y_train)

# 6. Evaluate the Models
print("--- SVM ---")
print(f"Train Accuracy: {accuracy_score(clf_svm.predict(X_train), Y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(clf_svm.predict(X_test), Y_test):.4f}")

print("\n--- Logistic Regression ---")
print(f"Train Accuracy: {accuracy_score(clf_lr.predict(X_train), Y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(clf_lr.predict(X_test), Y_test):.4f}")

print("\n--- Random Forest ---")
print(f"Train Accuracy: {accuracy_score(clf_rf.predict(X_train), Y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(clf_rf.predict(X_test), Y_test):.4f}")

# 7. Save the models and scaler
with open('model_svm.pkl', 'wb') as f:
    pickle.dump(clf_svm, f)

with open('model_lr.pkl', 'wb') as f:
    pickle.dump(clf_lr, f)

with open('model_rf.pkl', 'wb') as f:
    pickle.dump(clf_rf, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

import os
# Remove old model.pkl if it exists to avoid confusion
if os.path.exists('model.pkl'):
    os.remove('model.pkl')

print("\nSaved models to model_svm.pkl, model_lr.pkl, model_rf.pkl and scaler to scaler.pkl")
