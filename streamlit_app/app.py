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
# PREMIUM STARTUP CSS
# --------------------------------------------------
st.markdown("""
<style>

.main {
    background: radial-gradient(circle at top left, #1e1b4b, #020617);
}

.hero {
    text-align: center;
    padding-top: 160px;
    padding-bottom: 140px;
}

.hero-title {
    font-size: 70px;
    font-weight: 800;
    color: white;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 22px;
    color: #94a3b8;
    margin-top: 25px;
    line-height: 1.6;
}

.stButton>button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    padding: 16px 45px;
    font-size: 18px;
    border-radius: 14px;
    border: none;
    margin-top: 50px;
    transition: 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
}

/* GLASS PANEL */
.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(18px);
    padding: 35px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 30px;
}

/* AI SCORE CARD */
.score-card {
    text-align: center;
    padding: 40px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.15));
    border: 1px solid rgba(255,255,255,0.1);
}

.score-value {
    font-size: 60px;
    font-weight: 800;
    color: white;
}

.score-label {
    color: #cbd5e1;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LANDING STATE
# --------------------------------------------------
if "start" not in st.session_state:
    st.session_state.start = False

if not st.session_state.start:

    st.markdown("""
    <div class="hero">
        <div class="hero-title">🌍 Tourism Intelligence AI</div>
        <div class="hero-subtitle">
            Predict user satisfaction.<br>
            Discover high-impact attractions.<br>
            Make smarter tourism decisions.
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

    col1, col2 = st.columns([1,1])

    # ---------------- Prediction Panel ----------------
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

        if st.button("Generate AI Prediction"):
            user_input = np.array([[visit_year, visit_month, visit_mode_encoded]])
            user_input_scaled = scaler.transform(user_input)
            prediction = model.predict(user_input_scaled)[0]

            st.markdown(f"""
            <div class="score-card">
                <div class="score-value">⭐ {prediction:.2f}</div>
                <div class="score-label">Predicted Satisfaction Score</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Recommendation Panel ----------------
    with col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Smart Attraction Recommendations")

        rec = (
            transaction[transaction["VisitModeLabel"] == visit_mode_label]
            .groupby("AttractionId")["Rating"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )

        if rec.empty:
            st.info("Not enough data.")
        else:
            fig = px.bar(
                rec,
                x="Rating",
                y="AttractionId",
                orientation="h",
                title="Top Attractions by Average Rating",
                color="Rating",
                color_continuous_scale="Purples"
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

