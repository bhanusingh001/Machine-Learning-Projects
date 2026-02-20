"""
Rock vs Mine Prediction - Streamlit Web App
=============================================
A premium, visually stunning web app that predicts whether a sonar signal
bounced off a Rock or a Mine using a trained Logistic Regression model.
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ─── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Rock vs Mine Predictor",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Font Import ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global Styles ── */
    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* ── Hero Section ── */
    .hero-container {
        text-align: center;
        padding: 2rem 1rem 1.5rem;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #a8b2d1;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea33, #764ba233);
        border: 1px solid #667eea55;
        padding: 0.35rem 1rem;
        border-radius: 50px;
        color: #ccd6f6;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.8rem;
        letter-spacing: 1px;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.8rem;
        margin: 0.8rem 0;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(102, 126, 234, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
    }
    .glass-card h3 {
        color: #ccd6f6;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    .glass-card p {
        color: #8892b0;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .card-icon {
        font-size: 2.2rem;
        margin-bottom: 0.6rem;
        display: block;
    }

    /* ── Result Cards ── */
    .result-rock {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.12), rgba(16, 185, 129, 0.08));
        border: 1.5px solid rgba(34, 197, 94, 0.35);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        animation: slideUp 0.5s ease-out;
    }
    .result-mine {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(220, 38, 127, 0.08));
        border: 1.5px solid rgba(239, 68, 68, 0.35);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        animation: slideUp 0.5s ease-out;
    }
    .result-emoji {
        font-size: 4rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .result-rock .result-label { color: #22c55e; }
    .result-mine .result-label { color: #ef4444; }
    .result-desc {
        color: #a8b2d1;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Section Titles ── */
    .section-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #ccd6f6;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .section-desc {
        color: #8892b0;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        color: #8892b0;
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.2rem;
    }

    /* ── Divider ── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea44, transparent);
        margin: 2rem 0;
        border: none;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ── Streamlit Overrides ── */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #ccd6f6 !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        font-size: 0.85rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.15) !important;
    }
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #ccd6f6 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2.5rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }
    .stRadio > div {
        color: #ccd6f6 !important;
    }

    /* ── Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 0.3rem;
        gap: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #8892b0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea33, #764ba233) !important;
        color: #ccd6f6 !important;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Info Box ── */
    .info-box {
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: #a8b2d1;
        font-size: 0.88rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model, Scaler & Data ───────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
    with open(scaler_path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "sonar data.csv")
    df = pd.read_csv(data_path, header=None)
    return df

model = load_model()
scaler = load_scaler()
sonar_data = load_data()

# Get sample data
rock_samples = sonar_data[sonar_data[60] == "R"]
mine_samples = sonar_data[sonar_data[60] == "M"]

# ─── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <span style="font-size: 3rem;">🔊</span>
        <h2 style="color: #ccd6f6; font-weight: 800; margin: 0.3rem 0 0;">Sonar AI</h2>
        <p style="color: #8892b0; font-size: 0.85rem;">Rock vs Mine Classifier</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">208</div>
        <div class="metric-label">Total Samples</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(rock_samples)}</div>
            <div class="metric-label">Rock Samples</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(mine_samples)}</div>
            <div class="metric-label">Mine Samples</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">60</div>
        <div class="metric-label">Sonar Features</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>About the Model</strong><br>
        Logistic Regression trained on the UCI Sonar dataset.
        Each sample has 60 frequency band energy values (0.0 - 1.0)
        from sonar signals bounced off different surfaces.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center;">
        <a href="https://github.com/bsingh0001" target="_blank"
           style="color: #667eea; text-decoration: none; font-weight: 600; font-size: 0.9rem;">
           ⭐ Star on GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)


# ─── Hero Section ────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Rock vs Mine Predictor</h1>
    <p class="hero-subtitle">
        Classify sonar signals using Machine Learning — Instantly detect underwater objects
    </p>
    <span class="hero-badge">🤖 POWERED BY LOGISTIC REGRESSION</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ─── How It Works ────────────────────────────────────────────────────
st.markdown('<p class="section-title">⚡ How It Works</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="glass-card">
        <span class="card-icon">🔊</span>
        <h3>1. Sonar Signal</h3>
        <p>A sonar pulse is emitted and bounces off an underwater object. The returning signal is captured across 60 frequency bands.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
        <span class="card-icon">🧠</span>
        <h3>2. ML Analysis</h3>
        <p>Our Logistic Regression model analyzes the 60 frequency band energy values to find patterns unique to rocks and mines.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="glass-card">
        <span class="card-icon">🎯</span>
        <h3>3. Prediction</h3>
        <p>The model classifies the signal as either a Rock (natural formation) or a Mine (metal cylinder), helping naval operations.</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ─── Prediction Section ──────────────────────────────────────────────
st.markdown('<p class="section-title">🎯 Make a Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="section-desc">Choose an input method below to classify a sonar signal</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📋  Use Sample Data", "✏️  Enter Manual Values"])

with tab1:
    st.markdown("""
    <div class="info-box">
        <strong>How this works:</strong> The dataset has 97 Rock samples and 111 Mine samples.
        Pick a category below, then choose which sample number you want to test.
        The model will predict what it thinks the object is — then you can see if it got it right!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Step 1: Choose category ──
    sample_type = st.selectbox(
        "Step 1 — What type of sample do you want to test?",
        ["🪨 Rock Sample", "💣 Mine Sample", "🎲 Random (Unknown)"],
        key="sample_type"
    )

    # ── Step 2: Choose which one ──
    if "Rock" in sample_type:
        total = len(rock_samples)
        label_text = f"Step 2 — Pick a Rock sample (1 to {total})"
    elif "Mine" in sample_type:
        total = len(mine_samples)
        label_text = f"Step 2 — Pick a Mine sample (1 to {total})"
    else:
        total = len(sonar_data)
        label_text = f"Step 2 — Pick any sample (1 to {total})"

    sample_num = st.slider(
        label_text,
        min_value=1,
        max_value=total,
        value=1,
        key="sample_num",
        help="Each number represents a different sonar reading from the dataset. Try different numbers to test the model on various samples!"
    )
    sample_idx = sample_num - 1  # convert to 0-indexed

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Predict This Sample", key="predict_sample"):
        if "Rock" in sample_type:
            input_data = rock_samples.iloc[sample_idx, :60].values
            actual = "Rock"
        elif "Mine" in sample_type:
            input_data = mine_samples.iloc[sample_idx, :60].values
            actual = "Mine"
        else:
            input_data = sonar_data.iloc[sample_idx, :60].values
            actual = "Rock" if sonar_data.iloc[sample_idx, 60] == "R" else "Mine"

        input_reshaped = input_data.reshape(1, -1)
        input_scaled = scaler.transform(input_reshaped)
        prediction = model.predict(input_scaled)[0]

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == "R":
            st.markdown("""
            <div class="result-rock">
                <span class="result-emoji">🪨</span>
                <span class="result-label">ROCK DETECTED</span>
                <p class="result-desc">The sonar signal bounced off a natural rock formation on the ocean floor.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-mine">
                <span class="result-emoji">💣</span>
                <span class="result-label">MINE DETECTED</span>
                <p class="result-desc">The sonar signal bounced off a metal cylinder (mine) — proceed with caution!</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        actual_emoji = "🪨" if actual == "Rock" else "💣"
        pred_label = "Rock" if prediction == "R" else "Mine"
        match = "✅ Correct!" if (pred_label == actual) else "❌ Incorrect"

        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{pred_label}</div>
                <div class="metric-label">Predicted</div>
            </div>
            """, unsafe_allow_html=True)
        with rcol2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{actual}</div>
                <div class="metric-label">Actual</div>
            </div>
            """, unsafe_allow_html=True)
        with rcol3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{match}</div>
                <div class="metric-label">Result</div>
            </div>
            """, unsafe_allow_html=True)


with tab2:
    st.markdown("""
    <div class="info-box">
        Enter 60 comma-separated values (sonar frequency band energies between 0.0 and 1.0).
        Each value represents the energy in a specific frequency band of the sonar return signal.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Default sample values
    default_rock = ",".join([str(v) for v in rock_samples.iloc[0, :60].values])
    default_mine = ",".join([str(v) for v in mine_samples.iloc[0, :60].values])

    quick_fill = st.radio(
        "Quick fill with sample data:",
        ["Empty", "Rock sample", "Mine sample"],
        horizontal=True,
        key="quick_fill"
    )

    if quick_fill == "Rock sample":
        default_val = default_rock
    elif quick_fill == "Mine sample":
        default_val = default_mine
    else:
        default_val = ""

    user_input = st.text_area(
        "Paste your 60 sonar values here:",
        value=default_val,
        height=120,
        placeholder="0.0200, 0.0371, 0.0428, 0.0207, ... (60 values)",
        key="manual_input"
    )

    if st.button("🔍 Predict Values", key="predict_manual"):
        try:
            values = [float(x.strip()) for x in user_input.split(",") if x.strip()]

            if len(values) != 60:
                st.error(f"⚠️ Expected 60 values but got {len(values)}. Please enter exactly 60 comma-separated numbers.")
            else:
                input_array = np.array(values).reshape(1, -1)
                input_scaled = scaler.transform(input_array)
                prediction = model.predict(input_scaled)[0]

                st.markdown("<br>", unsafe_allow_html=True)

                if prediction == "R":
                    st.markdown("""
                    <div class="result-rock">
                        <span class="result-emoji">🪨</span>
                        <span class="result-label">ROCK DETECTED</span>
                        <p class="result-desc">The sonar signal bounced off a natural rock formation on the ocean floor.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-mine">
                        <span class="result-emoji">💣</span>
                        <span class="result-label">MINE DETECTED</span>
                        <p class="result-desc">The sonar signal bounced off a metal cylinder (mine) — proceed with caution!</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Show input statistics
                st.markdown("<br>", unsafe_allow_html=True)
                scol1, scol2, scol3, scol4 = st.columns(4)
                with scol1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{np.mean(values):.4f}</div>
                        <div class="metric-label">Mean Energy</div>
                    </div>
                    """, unsafe_allow_html=True)
                with scol2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{np.max(values):.4f}</div>
                        <div class="metric-label">Max Energy</div>
                    </div>
                    """, unsafe_allow_html=True)
                with scol3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{np.min(values):.4f}</div>
                        <div class="metric-label">Min Energy</div>
                    </div>
                    """, unsafe_allow_html=True)
                with scol4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{np.std(values):.4f}</div>
                        <div class="metric-label">Std Dev</div>
                    </div>
                    """, unsafe_allow_html=True)

        except ValueError:
            st.error("⚠️ Invalid input! Please enter numeric values separated by commas.")


st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ─── About Section ───────────────────────────────────────────────────
st.markdown('<p class="section-title">📊 Model Performance</p>', unsafe_allow_html=True)

perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

with perf_col1:
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <div class="metric-value" style="font-size: 2.5rem;">83%</div>
        <div class="metric-label">Test Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with perf_col2:
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <div class="metric-value" style="font-size: 2.5rem;">LR</div>
        <div class="metric-label">Algorithm</div>
    </div>
    """, unsafe_allow_html=True)

with perf_col3:
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <div class="metric-value" style="font-size: 2.5rem;">60</div>
        <div class="metric-label">Features</div>
    </div>
    """, unsafe_allow_html=True)

with perf_col4:
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <div class="metric-value" style="font-size: 2.5rem;">208</div>
        <div class="metric-label">Samples</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ─── Input Guide ─────────────────────────────────────────────────────
st.markdown('<p class="section-title">📖 Understanding the Input</p>', unsafe_allow_html=True)

with st.expander("What are the 60 sonar values?", expanded=False):
    st.markdown("""
    Each sonar sample consists of **60 numerical values** between 0.0 and 1.0:

    | Feature | Description |
    |---------|-------------|
    | **Values 1-60** | Energy within a specific frequency band over a set period of time |
    | **Range** | 0.0 (no energy) to 1.0 (maximum energy) |
    | **Signal Source** | Sonar signals bounced off either rocks or metal cylinders (mines) |

    **How sonar works:**
    1. A sonar transmitter sends an acoustic pulse underwater
    2. The pulse bounces off an object and returns to the receiver
    3. The returned signal is analyzed across 60 frequency bands
    4. Rocks and mines produce different frequency patterns due to their material properties

    **Rock signals** tend to have more scattered, irregular energy patterns.
    **Mine signals** often show stronger, more concentrated energy in certain bands.
    """)

with st.expander("Sample input format", expanded=False):
    st.code(
        ",".join([str(v) for v in rock_samples.iloc[0, :60].values]),
        language=None
    )
    st.markdown("*Copy the above line and paste it into the manual input field to test!*")


# ─── Footer ──────────────────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 1rem 0 2rem;">
    <p style="color: #4a5568; font-size: 0.85rem; font-weight: 400;">
        Built with ❤️ using Streamlit & Scikit-learn
        <br>
        <span style="color: #667eea;">Rock vs Mine Prediction</span> • ML Project by Bhanu
    </p>
</div>
""", unsafe_allow_html=True)
