"""
Diabetes Prediction - Streamlit Web App
=============================================
A premium, visually stunning web app that predicts whether an individual has
Diabetes using an ensemble of Machine Learning models. Features a light blue, modern UI,
Model selection, Risk Probability Meter, and a Health Intelligence Dashboard.
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ─── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Predictor Pro",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS for Light Blue UI ────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Font Import ── */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global Styles ── */
    * { font-family: 'Outfit', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #f0f8ff 0%, #e0f2fe 50%, #bae6fd 100%);
        color: #1e293b;
    }

    /* ── Hero Section ── */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 2rem;
        background: rgba(255, 255, 255, 0.6);
        border-radius: 24px;
        box-shadow: 0 10px 30px rgba(14, 165, 233, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.8);
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #475569;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(2, 132, 199, 0.1);
        border: 1px solid rgba(2, 132, 199, 0.3);
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        color: #0369a1;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 1rem;
        letter-spacing: 1px;
    }

    /* ── Light Glass Card ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 2rem;
        margin: 0.8rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.5);
        transform: translateY(-4px);
        box-shadow: 0 12px 25px rgba(2, 132, 199, 0.15);
    }
    .glass-card h3 {
        color: #0c4a6e;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    .glass-card p {
        color: #334155;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
    }

    /* ── Risk Meter ── */
    .risk-high { color: #dc2626; font-weight: 800; font-size: 3rem; margin-top: -10px;}
    .risk-medium { color: #f59e0b; font-weight: 800; font-size: 3rem; margin-top: -10px;}
    .risk-low { color: #16a34a; font-weight: 800; font-size: 3rem; margin-top: -10px;}
    
    .progress-bg {
        width: 100%;
        background-color: #e2e8f0;
        border-radius: 50px;
        height: 24px;
        margin-top: 15px;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        border-radius: 50px;
        transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* ── Result Cards ── */
    .result-card {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
    }
    .result-positive { border-color: rgba(239, 68, 68, 0.4); background: linear-gradient(135deg, rgba(239, 68, 68, 0.05), #ffffff); }
    .result-negative { border-color: rgba(34, 197, 94, 0.4); background: linear-gradient(135deg, rgba(34, 197, 94, 0.05), #ffffff); }

    .result-emoji { font-size: 4.5rem; display: block; margin-bottom: 0.5rem; }
    .result-label { font-size: 2rem; font-weight: 800; letter-spacing: -0.5px; }
    .result-positive .result-label { color: #dc2626; }
    .result-negative .result-label { color: #16a34a; }
    .result-desc { color: #475569; font-size: 1.05rem; margin-top: 0.8rem; font-weight: 500; }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Section Titles ── */
    .section-title { font-size: 1.8rem; font-weight: 800; color: #0f172a; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.6rem; }
    .section-desc { color: #64748b; font-size: 1rem; margin-bottom: 1.5rem; font-weight: 400; }

    /* ── Metric Cards ── */
    .metric-card {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value { font-size: 1.4rem; font-weight: 800; color: #0284c7; white-space: nowrap;}
    .metric-label { color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.4rem; }

    /* ── Divider ── */
    .custom-divider { height: 2px; background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.3), transparent); margin: 2.5rem 0; border: none; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; }

    /* ── Streamlit Overrides ── */
    .stTextArea textarea { background: #ffffff !important; border: 2px solid #cbd5e1 !important; border-radius: 12px !important; color: #334155 !important; font-family: 'Fira Code', monospace !important; font-size: 0.9rem !important; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02) !important; }
    .stTextArea textarea:focus { border-color: #0ea5e9 !important; box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.2) !important; }
    .stSelectbox > div > div { background: #ffffff !important; border: 2px solid #cbd5e1 !important; border-radius: 12px !important; color: #1e293b !important; font-weight: 600;}
    .stSelectbox > div > div:focus-within { border-color: #0ea5e9 !important; box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.2) !important; }
    .stButton > button { background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 0.8rem 2.5rem !important; font-weight: 700 !important; font-size: 1.1rem !important; letter-spacing: 0.5px !important; transition: all 0.3s ease !important; width: 100%; box-shadow: 0 4px 6px rgba(14, 165, 233, 0.2) !important; }
    .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 20px rgba(14, 165, 233, 0.3) !important; }
    
    /* ── Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] { background: rgba(255, 255, 255, 0.5); border-radius: 12px; padding: 0.4rem; gap: 0.4rem; border: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #64748b; font-weight: 600; padding: 0.5rem 1rem; }
    .stTabs [aria-selected="true"] { background: white !important; color: #0284c7 !important; box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important; }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Info Box ── */
    .info-box { background: #f0f9ff; border-left: 4px solid #0ea5e9; border-radius: 0 12px 12px 0; padding: 1.2rem 1.5rem; color: #0f172a; font-size: 0.95rem; line-height: 1.6; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    .info-box strong { color: #0369a1; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models, Scaler & Data ───────────────────────────────────────
@st.cache_resource
def load_assets():
    models = {
        "Random Forest (Highest Accuracy)": pickle.load(open(os.path.join(os.path.dirname(__file__), "model_rf.pkl"), "rb")),
        "Support Vector Machine (Balanced)": pickle.load(open(os.path.join(os.path.dirname(__file__), "model_svm.pkl"), "rb")),
        "Logistic Regression (Baseline)": pickle.load(open(os.path.join(os.path.dirname(__file__), "model_lr.pkl"), "rb"))
    }
    scaler = pickle.load(open(os.path.join(os.path.dirname(__file__), "scaler.pkl"), "rb"))
    return models, scaler

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "diabetes.csv"))
    return df

try:
    models, scaler = load_assets()
    diabetes_data = load_data()
    
    # Dataset stats
    diabetic_samples = diabetes_data[diabetes_data['Outcome'] == 1]
    non_diabetic_samples = diabetes_data[diabetes_data['Outcome'] == 0]
    total_samples = len(diabetes_data)
    
    # Means and Medians for comparison
    dataset_median = diabetes_data.median()
    
except Exception as e:
    st.error(f"Error loading files. Run 'python diabetes_model.py' first. Error: {e}")
    st.stop()

# ─── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <span style="font-size: 3.5rem;">🧠</span>
        <h2 style="color: #0f172a; font-weight: 800; margin: 0.5rem 0 0;">Med-AI Pro</h2>
        <p style="color: #64748b; font-size: 0.9rem;">Multi-Model Diagnostic Tool</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider" style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)

    # 🎀 Model Selection 
    st.markdown("<p style='font-weight:700; color:#0f172a; margin-bottom:5px;'>1. Select AI Brain</p>", unsafe_allow_html=True)
    selected_model_name = st.selectbox("Choose the algorithm:", list(models.keys()), label_visibility="collapsed")
    active_model = models[selected_model_name]

    st.markdown('<div class="custom-divider" style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="border-left-color: #10b981; border-radius: 8px;">
        <strong>Ensemble System Active</strong><br>
        Models calculate a raw probability score (0-100%).
        A score over 50% triggers a high-risk warning.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center;">
        <a href="https://github.com/bhanusingh001" target="_blank"
           style="color: #0284c7; text-decoration: none; font-weight: 700; font-size: 0.95rem;">
           ⭐ View on GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)


# ─── Hero Section ────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Diabetes Health Intelligence</h1>
    <p class="hero-subtitle">
        Comparing multiple AI models for accurate health predictions
    </p>
    <span class="hero-badge">🧬 MACHINE LEARNING DIAGNOSTICS</span>
</div>
""", unsafe_allow_html=True)

# ─── Helper Functions ────────────────────────────────────────────────
def render_risk_meter(probability):
    prob_percent = probability * 100
    if prob_percent >= 60:
        color_class = "risk-high"
        bg_color = "#dc2626"
        label = "HIGH RISK"
    elif prob_percent >= 40:
        color_class = "risk-medium"
        bg_color = "#f59e0b"
        label = "MODERATE RISK"
    else:
        color_class = "risk-low"
        bg_color = "#16a34a"
        label = "LOW RISK"

    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 1rem;">
        <p style="color: #64748b; font-weight: 600; font-size: 1rem; margin-bottom: 0;">AI Risk Confidence</p>
        <div class="{color_class}">{prob_percent:.1f}%</div>
        <p style="color: {bg_color}; font-weight: 700; font-size: 1.2rem; margin-top: -5px; letter-spacing: 1px;">{label}</p>
        <div class="progress-bg">
            <div class="progress-bar" style="width: {prob_percent}%; background-color: {bg_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_ai_opinion(input_scaled):
    st.markdown('<br><p class="section-title">🧠 AI Multi-Model Consensus</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Comparing risk assessments from all three specialized "AI Brains".</p>', unsafe_allow_html=True)
    
    # Calculate probabilities for all models
    model_names = []
    probs = []
    
    for name, model in models.items():
        short_name = name.split(' (')[0]
        # Get probability of class 1 (Diabetic)
        p = model.predict_proba(input_scaled)[0][1] * 100
        # Ensure a tiny value so the bar is visible even if 0
        model_names.append(short_name)
        probs.append(max(p, 0.5)) 
    
    # Create DataFrame
    df_opinion = pd.DataFrame({
        'Model': model_names,
        'Confidence (%)': probs
    }).set_index('Model')
    
    # Display the chart
    st.bar_chart(df_opinion, y="Confidence (%)", color="#0284c7")

def render_health_dashboard(values):
    st.markdown('<br><p class="section-title">📊 Patient Health Benchmark</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Comparing patient metrics against the global dataset median.</p>', unsafe_allow_html=True)
    
    # We compare 4 main features: Glucose, BMI, Age, BloodPressure
    comp_df = pd.DataFrame({
        'Metric': ['Glucose', 'BMI', 'Age', 'Blood Pressure'],
        'Patient Value': [values[1], values[5], values[7], values[2]],
        'Dataset Median': [dataset_median['Glucose'], dataset_median['BMI'], dataset_median['Age'], dataset_median['BloodPressure']]
    }).set_index('Metric')
    
    st.bar_chart(comp_df, color=["#0ea5e9", "#94a3b8"])


# ─── Prediction Section ──────────────────────────────────────────────
st.markdown('<p class="section-title">🎯 Clinical Assessment</p>', unsafe_allow_html=True)
st.markdown('<p class="section-desc">Test historical records or input custom patient data</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📋  Test Historical Records", "✏️  Enter Custom Patient Data"])

with tab1:
    st.markdown(f"""
    <div class="info-box">
        <strong>Validation Mode:</strong> Verify the effectiveness of <b>{selected_model_name}</b> against {total_samples} historical records.
    </div>
    <br>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-weight:700; color:#0f172a;'>Browse Patient Data (Full Dataset):</p>", unsafe_allow_html=True)
    total = total_samples
    sample_num = st.slider(f"Select Patient Record Index (1 to {total}):", min_value=1, max_value=total, value=1, key="sample_num")
    sample_idx = sample_num - 1

    if st.button("🔍 Run Full Diagnostics", key="predict_sample"):
        input_data = diabetes_data.iloc[sample_idx, :8].values
        actual = "Diabetic" if diabetes_data.iloc[sample_idx, 8] == 1 else "Non-Diabetic"

        input_reshaped = input_data.reshape(1, -1)
        input_scaled = scaler.transform(input_reshaped)
        
        # Get Probabilities
        probabilities = active_model.predict_proba(input_scaled)[0]
        diabetic_prob = probabilities[1]
        prediction = 1 if diabetic_prob >= 0.5 else 0

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display Results
        res_col1, res_col2 = st.columns([1, 1.2])

        with res_col1:
            st.markdown(f"<div class='result-card {'result-positive' if prediction==1 else 'result-negative'}'>", unsafe_allow_html=True)
            if prediction == 1:
                st.markdown("<span class='result-emoji'>⚠️</span><span class='result-label'>DIABETIC PROFILE</span>", unsafe_allow_html=True)
                st.markdown("<p class='result-desc'>Machine Learning model detects patterns matching Diabetes.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='result-emoji'>✅</span><span class='result-label'>HEALTHY PROFILE</span>", unsafe_allow_html=True)
                st.markdown("<p class='result-desc'>No significant diabetic patterns detected by AI.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with res_col2:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            render_risk_meter(diabetic_prob)
            st.markdown("</div>", unsafe_allow_html=True)

        # Validation Metrics
        st.markdown("<br>", unsafe_allow_html=True)
        pred_label = "Diabetic" if prediction == 1 else "Non-Diabetic"
        match = "✔️ AI is Correct" if (pred_label == actual) else "❌ AI Made a Mistake"

        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: {"#dc2626" if prediction==1 else "#16a34a"};">{pred_label}</div><div class="metric-label">Predicted</div></div>', unsafe_allow_html=True)
        with rcol2:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: {"#dc2626" if actual=="Diabetic" else "#16a34a"};">{actual}</div><div class="metric-label">Actual True Diagnosis</div></div>', unsafe_allow_html=True)
        with rcol3:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: {"#16a34a" if "Correct" in match else "#dc2626"};">{match}</div><div class="metric-label">Validation Tool</div></div>', unsafe_allow_html=True)

        # Dashboard & AI Consensus
        render_ai_opinion(input_scaled)
        render_health_dashboard(input_data)


with tab2:
    st.markdown("""
    <div class="info-box">
        <strong>Manual Entry:</strong> Enter 8 exact clinical parameters in order: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
    </div>
    <br>
    """, unsafe_allow_html=True)

    default_diabetic = ",".join([str(v) for v in diabetic_samples.iloc[0, :8].values])
    default_healthy = ",".join([str(v) for v in non_diabetic_samples.iloc[0, :8].values])
    quick_fill = st.radio("Auto-fill templates:", ["Empty Form", "Known Diabetic Trajectory", "Known Healthy Trajectory"], horizontal=True)

    default_val = default_diabetic if "Diabetic" in quick_fill else default_healthy if "Healthy" in quick_fill else ""

    user_input = st.text_area("Enter 8 comma-separated numerical values:", value=default_val, height=100, placeholder="e.g., 6, 148, 72, 35, 0, 33.6, 0.627, 50")

    if st.button("🔍 Run Full Diagnostics", key="predict_manual"):
        try:
            values = [float(x.strip()) for x in user_input.split(",") if x.strip()]

            if len(values) != 8:
                st.error(f"⚠️ Requires exactly 8 parameters. You provided {len(values)}.")
            else:
                input_array = np.array(values).reshape(1, -1)
                input_scaled = scaler.transform(input_array)
                
                # Predict probability
                probabilities = active_model.predict_proba(input_scaled)[0]
                diabetic_prob = probabilities[1]
                prediction = 1 if diabetic_prob >= 0.5 else 0

                st.markdown("<br>", unsafe_allow_html=True)
                res_col1, res_col2 = st.columns([1, 1.2])

                with res_col1:
                    st.markdown(f"<div class='result-card {'result-positive' if prediction==1 else 'result-negative'}'>", unsafe_allow_html=True)
                    if prediction == 1:
                        st.markdown("<span class='result-emoji'>⚠️</span><span class='result-label'>DIABETIC PROFILE</span>", unsafe_allow_html=True)
                        st.markdown("<p class='result-desc'>Machine Learning model detects patterns matching Diabetes.</p>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='result-emoji'>✅</span><span class='result-label'>HEALTHY PROFILE</span>", unsafe_allow_html=True)
                        st.markdown("<p class='result-desc'>No significant diabetic patterns detected by AI.</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with res_col2:
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    render_risk_meter(diabetic_prob)
                    st.markdown("</div>", unsafe_allow_html=True)

                # Dashboard & AI Consensus
                render_ai_opinion(input_scaled)
                render_health_dashboard(values)

        except ValueError:
            st.error("⚠️ Invalid input! Please enter numeric values separated by commas.")


st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 1rem 0 2rem;">
    <p style="color: #64748b; font-size: 0.9rem; font-weight: 500;">
        Built with ❤️ using Streamlit & Scikit-learn
        <br>
        <span style="color: #0284c7; font-weight: 700;">Multi-Model Architecture</span> • Professional Edition
    </p>
</div>
""", unsafe_allow_html=True)
