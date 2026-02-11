import streamlit as st
import pandas as pd
import numpy as np
import os
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
# MODERN AI PRODUCT CSS
# --------------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.main {
    background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
}

/* HERO SECTION */

.hero {
    text-align: center;
    padding-top: 120px;
    padding-bottom: 120px;
}

.hero-title {
    font-size: 64px;
    font-weight: 800;
    color: white;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 20px;
    color: #94a3b8;
    margin-top: 20px;
}

.stButton>button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    padding: 14px 40px;
    font-size: 18px;
    border-radius: 12px;
    border: none;
    margin-top: 40px;
    transition: 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
}

/* GLASS CARDS */

.glass {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(15px);
    padding: 30px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 25px;
}

.metric-value {
    font-size: 40px;
    font-weight: 700;
    color: #ffffff;
}

.metric-label {
    color: #94a3b8;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LANDING PAGE
# --------------------------------------------------

if "start" not in st.session_state:
    st.session_state.start = False

if not st.session_state.start:

    st.markdown("""
    <div class="hero">
        <div class="hero-title">🌍 Tourism Intelligence AI</div>
        <div class="hero-subtitle">
            Experience next-generation AI-powered tourism analytics.
            Predict satisfaction. Discover insights. Optimize decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Launch AI Engine"):
        st.session_state.start = True
        st.rerun()

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

else:

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

    le = LabelEncoder()
    transaction["VisitMode_encoded"] = le.fit_transform(transaction["VisitModeLabel"])

    X = transaction[["VisitYear", "VisitMonth", "VisitMode_encoded"]]
    y = transaction["Rating"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    st.markdown("## 🤖 AI Experience Engine")

    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Predict User Satisfaction")

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

            st.markdown(f"""
            <div class="metric-value">
                ⭐ {prediction:.2f} / 5
            </div>
            <div class="metric-label">
                AI Predicted Experience Score
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Smart Recommendations")

        rec = (
            transaction[transaction["VisitModeLabel"] == visit_mode_label]
            .groupby("AttractionId")["Rating"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )

        if rec.empty:
            st.info("Insufficient data available.")
        else:
            rec.columns = ["Attraction ID", "Average Rating"]
            st.dataframe(rec, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 📊 Intelligence Insights")
    st.bar_chart(transaction["Rating"].value_counts().sort_index())
