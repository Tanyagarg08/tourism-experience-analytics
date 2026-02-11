import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Tourism AI Engine", layout="wide")

# -----------------------------
# Custom Premium CSS
# -----------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f172a, #020617);
}

.big-title {
    font-size: 48px;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 30px;
}

.kpi-card {
    background: rgba(255,255,255,0.05);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
}

.ai-card {
    background: rgba(255,255,255,0.05);
    padding: 40px;
    border-radius: 25px;
    backdrop-filter: blur(12px);
}

.glow-button button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    font-weight: 600;
    border-radius: 30px;
    height: 50px;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_excel("data/Transaction.xlsx")

df = load_data()

# -----------------------------
# Landing Screen
# -----------------------------
if "launched" not in st.session_state:
    st.session_state.launched = False

if not st.session_state.launched:

    st.markdown('<div class="big-title">🌍 Tourism AI Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict • Analyze • Recommend</div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 Launch Intelligence Engine"):
            st.session_state.launched = True
            st.rerun()

# -----------------------------
# Dashboard
# -----------------------------
else:

    st.markdown('<div class="big-title">Tourism Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # KPI Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f'<div class="kpi-card"><h1>{round(df["Rating"].mean(),2)}</h1><p>Average Rating</p></div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="kpi-card"><h1>{len(df)}</h1><p>Total Visits</p></div>', unsafe_allow_html=True)

    with col3:
        st.markdown(f'<div class="kpi-card"><h1>{df["AttractionId"].nunique()}</h1><p>Unique Attractions</p></div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    left, right = st.columns([1,1])

    # ---------------- Prediction Panel ----------------
    with left:
        st.markdown("## 🎯 Experience Prediction")

        visit_year = st.number_input("Visit Year", min_value=2018, max_value=2025, value=2022)
        visit_month = st.slider("Visit Month", 1, 12, 6)
        visit_mode = st.selectbox("Visit Mode", sorted(df["VisitMode"].astype(str).unique()))

        le = LabelEncoder()
        df["VisitMode_encoded"] = le.fit_transform(df["VisitMode"].astype(str))

        X = df[["VisitYear", "VisitMonth", "VisitMode_encoded"]]
        y = df["Rating"]

        model = LinearRegression()
        model.fit(X, y)

        mode_encoded = le.transform([visit_mode])[0]
        prediction = model.predict([[visit_year, visit_month, mode_encoded]])[0]

        if st.button("Generate AI Prediction"):
            st.markdown(f"""
            <div class="ai-card">
                <h1>⭐ {round(prediction,2)} / 5</h1>
                <p>Predicted user satisfaction score</p>
            </div>
            """, unsafe_allow_html=True)

    # ---------------- Smart Analytics ----------------
    with right:
        st.markdown("## 🔥 Top Attractions")

        top = df.groupby("AttractionId")["Rating"].mean().sort_values(ascending=False).head(10).reset_index()

        fig = px.bar(
            top,
            x="Rating",
            y="AttractionId",
            orientation="h",
            color="Rating",
            color_continuous_scale="purples",
        )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(fig, use_container_width=True)
