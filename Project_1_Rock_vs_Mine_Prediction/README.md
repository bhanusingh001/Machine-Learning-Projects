# 🔊 Rock vs Mine Prediction

> **Classify underwater objects as Rocks or Mines using Sonar signals & Machine Learning**

[**🚀 View Live App**](https://rock-vs-mine-predictioonn.streamlit.app/)

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📋 Overview

This project uses **Logistic Regression** to classify sonar signals as either **Rock** 🪨 or **Mine** 💣. Sonar (Sound Navigation and Ranging) sends acoustic pulses underwater and analyzes the returned signals across **60 frequency bands** to determine whether the object is a natural rock formation or a dangerous metal mine.

The project includes a **premium Streamlit web app** with a dark glassmorphism UI for real-time predictions.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **ML Model** | Logistic Regression trained on the UCI Sonar Dataset |
| 🌐 **Web App** | Premium dark-themed Streamlit interface |
| 📊 **Dashboard** | Real-time model performance metrics |
| 🎯 **Dual Input** | Predict from sample data or enter custom values |
| 📱 **Responsive** | Works on desktop and mobile browsers |
| 🚀 **Deploy Ready** | One-click deployment to Streamlit Cloud |

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **ML Library:** Scikit-learn (Logistic Regression + StandardScaler)
- **Web Framework:** Streamlit
- **Data Processing:** NumPy, Pandas
- **Deployment:** Streamlit Cloud

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | [UCI Machine Learning Repository — Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)) |
| **Samples** | 208 (97 Rocks + 111 Mines) |
| **Features** | 60 (frequency band energy values, range: 0.0 — 1.0) |
| **Target** | R (Rock) or M (Mine) |

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Training Accuracy** | 91.98% |
| **Test Accuracy** | 76.19% |
| **Algorithm** | Logistic Regression + StandardScaler |
| **Test Split** | 10% (stratified) |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/bhanusingh001/Machine-Learning-Projects.git
cd Rock-vs-Mine-Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional — pre-trained model included)

```bash
python rock_vs_mine_model.py
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

The model expects **60 comma-separated numerical values** (each between 0.0 and 1.0), representing the energy across 60 frequency bands of a sonar return signal.

### Example Rock Input:
```
0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032
```

### Example Mine Input:
```
0.0491,0.0279,0.0592,0.1270,0.1772,0.1908,0.2217,0.0768,0.1246,0.2028,0.0947,0.2497,0.2209,0.3195,0.3340,0.3323,0.2780,0.2975,0.2948,0.1729,0.3264,0.3834,0.3523,0.5410,0.5228,0.4475,0.5340,0.5323,0.3907,0.3456,0.4091,0.4639,0.5580,0.5727,0.6355,0.7563,0.6903,0.6176,0.5379,0.5622,0.6508,0.4797,0.3736,0.2804,0.1982,0.2438,0.1789,0.1706,0.0762,0.0238,0.0268,0.0081,0.0129,0.0161,0.0063,0.0119,0.0194,0.0140,0.0332,0.0439
```

> **Tip:** In the Streamlit app, you can use the "Sample Data" tab to test with pre-loaded samples without typing anything!

---

## 📁 Project Structure

```
Rock-vs-Mine-Prediction/
├── app.py                  # Streamlit web application
├── rock_vs_mine_model.py   # Model training script
├── model.pkl               # Trained ML model
├── scaler.pkl              # Feature scaler (StandardScaler)
├── sonar data.csv          # Dataset
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## 🤝 Credits

- **Dataset:** UCI Machine Learning Repository
- **Tutorial Reference:** [Siddhardhan — ML Project Series](https://www.youtube.com/watch?v=fiz1ORTBGpY&list=PLfFghEzKVmjvuSA67LszN1dZ-Dd_pkus6)
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
