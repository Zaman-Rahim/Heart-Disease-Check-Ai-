import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HeartCheck AI", page_icon="❤️", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(135deg, #fdf2f8 0%, #eef2ff 50%, #ecfeff 100%); }
.hero {
    background: linear-gradient(120deg, #ef4444, #ec4899, #8b5cf6);
    padding: 2.2rem 2rem; border-radius: 22px; color: white;
    box-shadow: 0 12px 30px rgba(236, 72, 153, 0.30); margin-bottom: 1.5rem;
}
.hero h1 { margin: 0; font-size: 2.3rem; font-weight: 700; color: white; }
.hero p { margin: .4rem 0 0; opacity: .92; font-size: 1.02rem; }
.card-title { color: #4338ca; font-size: 1.15rem; font-weight: 600; margin: .2rem 0 .6rem; }
.result-high {
    background: linear-gradient(120deg, #fee2e2, #fecaca); border-left: 8px solid #ef4444;
    padding: 1.3rem; border-radius: 16px; color: #7f1d1d;
}
.result-low {
    background: linear-gradient(120deg, #dcfce7, #bbf7d0); border-left: 8px solid #22c55e;
    padding: 1.3rem; border-radius: 16px; color: #14532d;
}
.bar-wrap { background: #e5e7eb; border-radius: 999px; height: 16px; overflow: hidden; margin-top: .6rem; }
.bar { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #22c55e, #facc15, #ef4444); }
div.stButton > button {
    width: 100%; border: none; border-radius: 14px; padding: .8rem 1rem; font-weight: 600; font-size: 1.05rem;
    color: white; background: linear-gradient(120deg, #6366f1, #ec4899);
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.35); transition: transform .15s;
}
div.stButton > button:hover { transform: translateY(-2px); color: white; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: white; border-radius: 18px; }
.disclaimer { font-size: .85rem; color: #6b7280; text-align: center; margin-top: 1.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Load model, scaler, training columns ----------
NUM_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]


@st.cache_resource
def load_artifacts():
    return (
        joblib.load("heart_model.pkl"),
        joblib.load("scaler.pkl"),
        joblib.load("columns.pkl"),
    )


try:
    model, scaler, train_cols = load_artifacts()
except Exception as e:
    st.error(
        f"Could not load model files (heart_model.pkl, scaler.pkl, columns.pkl): {e}"
    )
    st.stop()


def run_model(raw: pd.DataFrame):
    """Repeat exactly the preprocessing used in training:
    one-hot encode -> align to training columns -> scale numeric columns."""
    enc = pd.get_dummies(raw, dtype=int)
    # Dropped baseline categories (drop_first=True) are simply discarded here,
    # and any category not present in this single row is filled with 0.
    enc = enc.reindex(columns=train_cols, fill_value=0)
    enc[NUM_COLS] = scaler.transform(enc[NUM_COLS])
    pred = int(model.predict(enc)[0])
    prob = float(model.predict_proba(enc)[0][1]) if hasattr(model, "predict_proba") else None
    return pred, prob


# ---------- Label -> dataset value ----------
SEX = {"Male": "M", "Female": "F"}
CP = {
    "Typical Angina": "TA",
    "Atypical Angina": "ATA",
    "Non-Anginal Pain": "NAP",
    "Asymptomatic": "ASY",
}
FBS = {"No (≤ 120 mg/dl)": 0, "Yes (> 120 mg/dl)": 1}
ECG = {"Normal": "Normal", "ST-T wave abnormality": "ST", "Left ventricular hypertrophy": "LVH"}
ANGINA = {"No": "N", "Yes": "Y"}
SLOPE = {"Upsloping": "Up", "Flat": "Flat", "Downsloping": "Down"}

# ---------- Header ----------
st.markdown(
    """
<div class="hero">
  <h1>❤️ HeartCheck AI by Rahim</h1>
  <p>Enter the patient details below to estimate heart disease risk using a machine learning model.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Inputs ----------
left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="card-title">👤 Personal Details</div>', unsafe_allow_html=True)
        age = st.slider("Age", 18, 100, 50)
        sex = st.radio("Sex", list(SEX), horizontal=True)
        cp = st.selectbox("Chest pain type", list(CP))

    with st.container(border=True):
        st.markdown('<div class="card-title">🩸 Blood & Vitals</div>', unsafe_allow_html=True)
        resting_bp = st.slider("Resting blood pressure (mm Hg)", 80, 220, 120)
        chol = st.slider("Serum cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.radio("Fasting blood sugar", list(FBS), horizontal=True)

with right:
    with st.container(border=True):
        st.markdown('<div class="card-title">📈 ECG & Exercise</div>', unsafe_allow_html=True)
        ecg = st.selectbox("Resting ECG result", list(ECG))
        max_hr = st.slider("Maximum heart rate achieved", 60, 220, 150)
        angina = st.radio("Exercise-induced angina", list(ANGINA), horizontal=True)
        oldpeak = st.slider("Oldpeak (ST depression)", -3.0, 7.0, 1.0, 0.1)
        slope = st.selectbox("ST slope (peak exercise)", list(SLOPE))

# ---------- Predict ----------
if st.button("🔍 Predict Risk"):
    X = pd.DataFrame(
        [
            {
                "Age": age,
                "Sex": SEX[sex],
                "ChestPainType": CP[cp],
                "RestingBP": resting_bp,
                "Cholesterol": chol,
                "FastingBS": FBS[fbs],
                "RestingECG": ECG[ecg],
                "MaxHR": max_hr,
                "ExerciseAngina": ANGINA[angina],
                "Oldpeak": oldpeak,
                "ST_Slope": SLOPE[slope],
            }
        ]
    )

    try:
        pred, prob = run_model(X)
    except Exception as e:
        st.error(
            "Prediction failed. The input format probably doesn't match how the model was trained "
            f"(encoding/column mismatch).\n\nDetails: {e}"
        )
        st.stop()

    st.markdown("### Result")
    if pred == 1:
        st.markdown(
            '<div class="result-high"><h3 style="margin:0">⚠️ Higher risk of heart disease detected</h3>'
            "<p style='margin:.4rem 0 0'>Please consult a qualified doctor for proper evaluation.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-low"><h3 style="margin:0">✅ Lower risk of heart disease</h3>'
            "<p style='margin:.4rem 0 0'>Keep up a healthy lifestyle and do regular check-ups.</p></div>",
            unsafe_allow_html=True,
        )

    if prob is not None:
        st.markdown(f"**Risk probability: {prob * 100:.1f}%**")
        st.markdown(
            f'<div class="bar-wrap"><div class="bar" style="width:{prob * 100:.0f}%"></div></div>',
            unsafe_allow_html=True,
        )

    with st.expander("See the input data sent to the model"):
        st.dataframe(X, use_container_width=True)

st.markdown(
    '<div class="disclaimer">⚠️ This tool is for educational/demo purposes only and is not a substitute '
    "for professional medical diagnosis.</div>",
    unsafe_allow_html=True,
)
