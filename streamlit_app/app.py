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
    page_title="Tourism Intelligence Dashboard",
    page_icon="🌍",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM STYLING (NO FAKE BUBBLES)
# --------------------------------------------------
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #0f172a, #020617);
}

.block-container {
    padding-top: 2rem;
}

/* Title */
.title {
    text-align: center;
    font-size: 50px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #94a3b8;
    margin-bottom: 40px;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}

.kpi-value {
    font-size: 36px;
    font-weight: 700;
    color: white;
}

.kpi-label {
    font-size: 14px;
    color: #94a3b8;
}

/* Section Box */
.section-box {
    background: #111827;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 20px;
}

.stButton>button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
    padding: 10px 30px;
    border: none;
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

visit_mode_map = {
    1: "Family",
    2: "Friends",
    3: "Couple",
    4: "Business"
}

transaction["VisitModeLabel"] = transaction["VisitMode"].map(visit_mode_map)
transaction.dropna(subset=["VisitModeLabel"], inplace=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown('<div class="title">🌍 Tourism Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-driven experience prediction & smart attraction insights</div>', unsafe_allow_html=True)

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{transaction['Rating'].mean():.2f}</div>
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
        <div class="kpi-value">{transaction['AttractionId'].nunique()}</div>
        <div class="kpi-label">Unique Attractions</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
le = LabelEncoder()
transaction["VisitMode_encoded"] = le.fit_transform(transaction["VisitModeLabel"])

X = transaction[["VisitYear", "VisitMonth", "VisitMode_encoded"]]
y = transaction["Rating"]

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --------------------------------------------------
# MAIN PANELS
# --------------------------------------------------
left, right = st.columns([1, 1])

# ---------------- AI Prediction ----------------
with left:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("🎯 AI Experience Prediction")

    visit_year = st.number_input("Visit Year", 2000, 2030, 2022)
    visit_month = st.slider("Visit Month", 1, 12, 6)
    visit_mode_label = st.selectbox(
        "Visit Mode",
        sorted(transaction["VisitModeLabel"].unique())
    )

    visit_mode_encoded = le.transform([visit_mode_label])[0]

    if st.button("Generate AI Prediction"):
        user_input = np.array([[visit_year, visit_month, visit_mode_encoded]])
        user_input_scaled = scaler.transform(user_input)
        prediction = model.predict(user_input_scaled)[0]

        st.metric("Predicted Satisfaction Score", f"{prediction:.2f} / 5")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Smart Recommendations ----------------
with right:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("🔥 Top Attractions For Selected Mode")

    rec = (
        transaction[transaction["VisitModeLabel"] == visit_mode_label]
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
        color_continuous_scale="Purples"
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)



