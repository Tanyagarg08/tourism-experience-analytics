import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Tourism Intelligence AI",
    page_icon="🌍",
    layout="wide"
)

# --------------------------------------------------
# CLEAN LIGHT THEME CSS
# --------------------------------------------------
st.markdown("""
<style>

/* Background */
.main {
    background-color: #ffffff;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headings */
h1, h2, h3 {
    color: #111827;
}

/* KPI Cards */
.kpi-card {
    background: #ffffff;
    padding: 28px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    text-align: center;
}

.kpi-value {
    font-size: 34px;
    font-weight: 700;
    color: #111827;
}

.kpi-label {
    font-size: 14px;
    color: #6b7280;
}

/* Section Panels */
.section-box {
    background: #ffffff;
    padding: 30px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}

/* Button Styling */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 12px 28px;
    border: none;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #1d4ed8;
}

/* Slider Accent */
.stSlider > div[data-baseweb="slider"] > div > div {
    background-color: #2563eb !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Transaction.xlsx")

@st.cache_data
def load_data(path):
    return pd.read_excel(path)

transaction = load_data(DATA_PATH)

# Map Visit Mode
visit_mode_map = {
    1: "Family",
    2: "Friends",
    3: "Couple",
    4: "Business"
}

transaction["VisitModeLabel"] = transaction["VisitMode"].map(visit_mode_map)
transaction.dropna(subset=["VisitModeLabel"], inplace=True)

# Encode
le = LabelEncoder()
transaction["VisitMode_encoded"] = le.fit_transform(transaction["VisitModeLabel"])

# Train Model
X = transaction[["VisitYear", "VisitMonth", "VisitMode_encoded"]]
y = transaction["Rating"]

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🌍 Tourism Intelligence Dashboard")
st.caption("AI-powered visitor experience prediction and attraction analytics")

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{transaction["Rating"].mean():.2f}</div>
        <div class="kpi-label">Average Rating</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{len(transaction)}</div>
        <div class="kpi-label">Total Visits</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{transaction["AttractionId"].nunique()}</div>
        <div class="kpi-label">Unique Attractions</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# MAIN PANELS
# --------------------------------------------------
left, right = st.columns([1,1])

# ---------------- Prediction Panel ----------------
with left:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("🤖 AI Experience Prediction")

    visit_year = st.number_input("Visit Year", 2000, 2030, 2022)
    visit_month = st.slider("Visit Month", 1, 12, 6)

    visit_mode_label = st.selectbox(
        "Visit Mode",
        sorted(transaction["VisitModeLabel"].unique())
    )

    visit_mode_encoded = le.transform([visit_mode_label])[0]

    if st.button("Generate Prediction"):
        user_input = np.array([[visit_year, visit_month, visit_mode_encoded]])
        user_input_scaled = scaler.transform(user_input)
        prediction = model.predict(user_input_scaled)[0]

        st.success(f"Predicted Satisfaction Score: ⭐ {prediction:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Recommendation Panel ----------------
with right:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("📊 Top Attractions by Visit Mode")

    rec = (
        transaction[transaction["VisitModeLabel"] == visit_mode_label]
        .groupby("AttractionId")["Rating"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )

    if rec.empty:
        st.info("Not enough data for selected visit mode.")
    else:
        fig = px.bar(
            rec,
            x="Rating",
            y="AttractionId",
            orientation="h",
            color="Rating",
            color_continuous_scale="Blues"
        )

        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="#111827",
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)