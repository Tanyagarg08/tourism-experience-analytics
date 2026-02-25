import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Tourism Intelligence",
    page_icon="🌍",
    layout="wide"
)

# --------------------------------------------------
# CLASSY LIGHT AI DASHBOARD THEME
# --------------------------------------------------
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #f6f9fc, #eef2f7);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.title {
    font-size: 40px;
    font-weight: 700;
    color: #1f2937;
}

.subtitle {
    font-size: 16px;
    color: #6b7280;
    margin-bottom: 25px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 6px 25px rgba(0,0,0,0.05);
}

.metric-title {
    font-size: 14px;
    color: #6b7280;
}

.metric-value {
    font-size: 28px;
    font-weight: 600;
    color: #111827;
}

.stButton>button {
    background: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 10px 30px;
    border: none;
}

.stButton>button:hover {
    background: #1d4ed8;
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

df = load_data(DATA_PATH)

# --------------------------------------------------
# CREATE VISIT MODE LABELS
# --------------------------------------------------
visit_mode_map = {
    1: "Family",
    2: "Friends",
    3: "Couple",
    4: "Business"
}

df["VisitModeLabel"] = df["VisitMode"].map(visit_mode_map)
df.dropna(subset=["VisitModeLabel"], inplace=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown('<div class="title">🌍 Tourism Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered visitor experience prediction and attraction analytics</div>', unsafe_allow_html=True)

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Average Rating</div>
        <div class="metric-value">{df['Rating'].mean():.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Total Visits</div>
        <div class="metric-value">{len(df)}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Unique Attractions</div>
        <div class="metric-value">{df['AttractionId'].nunique()}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# --------------------------------------------------
# MODEL
# --------------------------------------------------
le = LabelEncoder()
df["VisitMode_encoded"] = le.fit_transform(df["VisitModeLabel"])

X = df[["VisitYear", "VisitMonth", "VisitMode_encoded"]]
y = df["Rating"]

model = LinearRegression()
model.fit(X, y)

# --------------------------------------------------
# MAIN PANELS
# --------------------------------------------------
left, right = st.columns([1,1])

# ---------------- Prediction Panel ----------------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("AI Experience Prediction")

    visit_year = st.number_input("Visit Year", 2000, 2030, 2022)
    visit_month = st.slider("Visit Month", 1, 12, 6)
    visit_mode_label = st.selectbox(
        "Visit Mode",
        sorted(df["VisitModeLabel"].unique())
    )

    visit_mode_encoded = le.transform([visit_mode_label])[0]

    if st.button("Generate Prediction"):
        prediction = model.predict([[visit_year, visit_month, visit_mode_encoded]])[0]
        st.success(f"Predicted Satisfaction Score: ⭐ {prediction:.2f} / 5")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Recommendation Panel ----------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Top Attractions by Visit Mode")

    rec = (
        df[df["VisitModeLabel"] == visit_mode_label]
        .groupby("AttractionId")["Rating"]
        .mean()
        .sort_values(ascending=False)
        .head(7)
        .reset_index()
    )

    fig = px.bar(
        rec,
        x="Rating",
        y="AttractionId",
        orientation="h",
        color="Rating",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)