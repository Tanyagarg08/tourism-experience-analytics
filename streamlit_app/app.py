import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Global Styling (SAFE CSS)
# -------------------------------------------------
st.markdown(
    """
    <style>
        .section-card {
            background-color: #111827;
            padding: 24px;
            border-radius: 14px;
            margin-bottom: 24px;
            border: 1px solid #1f2937;
        }
        .result-card {
            background-color: #1f2937;
            padding: 26px;
            border-radius: 14px;
            border-left: 5px solid #6366f1;
        }
        .result-score {
            font-size: 36px;
            font-weight: 800;
            color: #f9fafb;
        }
        .result-text {
            margin-top: 8px;
            font-size: 15px;
            color: #d1d5db;
        }
        .confidence-text {
            font-size: 13px;
            color: #9ca3af;
            margin-top: 6px;
        }
        .title-text {
            font-size: 32px;
            font-weight: 800;
            color: #f9fafb;
        }
        .subtitle-text {
            font-size: 16px;
            color: #9ca3af;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    """
    <div style="margin-bottom:30px;">
        <div class="title-text">🌍 Tourism Experience Analytics</div>
        <div class="subtitle-text">
            Machine learning powered dashboard for tourism insights,
            experience prediction, and attraction recommendations.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Load Data
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Transaction.xlsx")

@st.cache_data
def load_data(path):
    return pd.read_excel(path)

transaction = load_data(DATA_PATH)

# -------------------------------------------------
# Visit Mode Mapping
# -------------------------------------------------
visit_mode_map = {
    1: "Family",
    2: "Friends",
    3: "Couple",
    4: "Business"
}

transaction["VisitModeLabel"] = transaction["VisitMode"].map(visit_mode_map)
transaction.dropna(subset=["VisitModeLabel"], inplace=True)

# -------------------------------------------------
# ML Preparation
# -------------------------------------------------
le = LabelEncoder()
transaction["VisitMode_encoded"] = le.fit_transform(transaction["VisitModeLabel"])

X = transaction[["VisitYear", "VisitMonth", "VisitMode_encoded"]]
y = transaction["Rating"]

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.markdown("## 🧑 User Inputs")
st.sidebar.caption("Adjust inputs to simulate a tourist visit")

visit_year = st.sidebar.number_input("Visit Year", 2000, 2030, 2022)
visit_month = st.sidebar.slider("Visit Month", 1, 12, 6)

visit_mode_label = st.sidebar.selectbox(
    "Visit Mode",
    sorted(transaction["VisitModeLabel"].unique())
)

visit_mode_encoded = le.transform([visit_mode_label])[0]

st.sidebar.markdown("---")
st.sidebar.info(
    "Predictions are generated using historical tourism data "
    "and a trained machine learning model."
)

# -------------------------------------------------
# Main Layout
# -------------------------------------------------
left, right = st.columns([1.3, 0.7])

# -------------------------------------------------
# Prediction Section
# -------------------------------------------------
with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### ⭐ Experience Rating Prediction")

    if st.button("Generate Experience Score"):
        user_input = np.array([[visit_year, visit_month, visit_mode_encoded]])
        user_input_scaled = scaler.transform(user_input)
        prediction = model.predict(user_input_scaled)[0]

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-score">⭐ {prediction:.2f} / 5</div>
                <div class="result-text">
                    Predicted user satisfaction score based on travel details.
                </div>
                <div class="confidence-text">
                    Model confidence derived from historical tourism patterns.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Recommended Attractions (SAFE HANDLING)
# -------------------------------------------------
with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Recommended Attractions")

    rec = (
        transaction[transaction["VisitModeLabel"] == visit_mode_label]
        .groupby("AttractionId")["Rating"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )

    if rec.empty:
        st.info("No sufficient data available for this visit mode.")
    else:
        rec.columns = ["Attraction ID", "Average Rating"]
        st.dataframe(rec, hide_index=True, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Charts (RESTORED)
# -------------------------------------------------
st.markdown("## 📊 Visual Insights")

c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### ⭐ Rating Distribution")
    st.bar_chart(transaction["Rating"].value_counts().sort_index())
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🧳 Visit Mode Popularity")
    st.bar_chart(transaction["VisitModeLabel"].value_counts())
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:14px; color:#9ca3af;">
        Internship-Ready End-to-End Machine Learning Dashboard
    </p>
    """,
    unsafe_allow_html=True
)
