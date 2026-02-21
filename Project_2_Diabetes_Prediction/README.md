# 🩸 Diabetes Prediction System

> **Predict the onset of Diabetes based on clinical diagnostic measures using Machine Learning**

[**🚀 View Live App**](https://your-streamlit-app-link-here.streamlit.app/) *(Add your Streamlit Cloud link here after deployment)*

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📋 Overview

This project uses an advanced **Multi-Model Machine Learning Ensemble** to predict whether an individual is at high risk of Diabetes based on 8 clinical diagnostic metrics. Instead of relying on a single algorithm, the system evaluates the patient using three distinct "AI Brains":

1.  **Support Vector Machine (SVM):** Provides a balanced, geometry-based decision boundary.
2.  **Random Forest Classifier:** Uses a voting system of decision trees for high accuracy and complex pattern recognition.
3.  **Logistic Regression:** Acts as a stable, statistical baseline.

By calculating a **Risk Probability** from all three models simultaneously, the application generates an **AI Consensus**, allowing medical professionals to see exactly how confident the algorithms are when assessing metrics like Glucose levels, BMI, Age, and Blood Pressure.

The project is presented through a **premium, sleek Streamlit web app** featuring a modern, light blue interface, designed for clarity and ease of use in medical contexts.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Multi-Model ML** | Ensemble of Random Forest, SVM, and Logistic Regression algorithms |
| 🌐 **Web App** | Premium, light blue modern Streamlit interface with Model Selector |
| 📊 **Pro Dashboard** | Patient Health Benchmark charts comparing data to the global dataset median |
| ⚖️ **AI Consensus** | Visual Risk Confidence meters (0-100%) showing agreement across all 3 models |
| 🎯 **Dual Input** | Test with historical sample data or evaluate custom patient cases |
| 📱 **Responsive** | Works flawlessly on desktop and mobile browsers |
| 🚀 **Deploy Ready** | One-click deployment to Streamlit Cloud |

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **ML Library:** Scikit-learn (RandomForestClassifier, SVC, LogisticRegression + StandardScaler)
- **Web Framework:** Streamlit
- **Data Processing:** NumPy, Pandas
- **Deployment:** Streamlit Cloud

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | [PIMA Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| **Samples** | 768 (268 Diabetic + 500 Healthy) |
| **Features** | 8 (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) |
| **Target** | 1 (Diabetic) or 0 (Non-Diabetic) |

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Best Model** | Random Forest Classifier |
| **RF Train Accuracy** | 100% |
| **RF Test Accuracy** | ~73% |
| **SVM/LR Accuracy** | ~76% - 78% |
| **Test Split** | 20% (stratified) |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/bhanusingh001/Machine-Learning-Projects.git
cd Project_2_Diabetes_Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional — pre-trained model included)

```bash
python diabetes_model.py
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

---

## 🌐 Deploy on Streamlit Cloud

1. Push this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your repo, branch (`main`), and file (`app.py`)
5. Click **"Deploy"** — your app will be live in minutes! 🚀

---

## 🔍 Input Format

The model expects **8 comma-separated numerical values**. 

### Important Parameters:
1. **Pregnancies:** Number of times pregnant
2. **Glucose:** Plasma glucose concentration (2 hours)
3. **BloodPressure:** Diastolic blood pressure (mm Hg)
4. **SkinThickness:** Triceps skin fold thickness (mm)
5. **Insulin:** 2-Hour serum insulin (mu U/ml)
6. **BMI:** Body mass index (weight in kg/(height in m)^2)
7. **DiabetesPedigreeFunction:** Genetic risk score
8. **Age:** Age (years)

### Example Diabetic Patient Input:
```
6, 148, 72, 35, 0, 33.6, 0.627, 50
```

### Example Healthy Patient Input:
```
1, 85, 66, 29, 0, 26.6, 0.351, 31
```

> **Tip:** In the Streamlit app, you can use the "Test Using Sample Records" tab to autofill the form with historical data!

---

## 📁 Project Structure

```text
Project_2_Diabetes_Prediction/
├── app.py                  # Streamlit web application
├── diabetes_model.py       # Script to train the 3-model ensemble
├── model_rf.pkl            # Trained Random Forest model
├── model_svm.pkl           # Trained Support Vector Machine model
├── model_lr.pkl            # Trained Logistic Regression model
├── scaler.pkl              # Feature scaler (StandardScaler)
├── diabetes.csv            # Dataset containing patient records
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🤝 Credits

- **Dataset:** PIMA Indians Diabetes Database (UCI Machine Learning Repository)
- **Tutorial Reference:** [Siddhardhan — ML Project Series](https://www.youtube.com/watch?v=xUE7SjVx9bQ&list=PLfFghEzKVmjvuSA67LszN1dZ-Dd_pkus6&index=2)
- **Built with:** [Streamlit](https://streamlit.io), [Scikit-learn](https://scikit-learn.org)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ by <b>Bhanu</b>
  <br>
  ⭐ Star this repo if you found it helpful!
</p>
