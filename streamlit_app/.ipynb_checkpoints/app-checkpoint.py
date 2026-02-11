import streamlit as st
import pandas as pd
import os

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Tourism Experience Analytics",
    layout="centered"
)

st.title("🌍 Tourism Experience Analytics")
st.write("This app analyzes tourism data and will predict ratings and visit modes.")

# -------------------------------
# Resolve data path safely
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Transaction.xlsx")

# -------------------------------
# Load data
# -------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    return df

# Debug / Info
st.write("📁 App running from:", BASE_DIR)
st.write("📄 Loading data from:", DATA_PATH)

# -------------------------------
# Load dataset
# -------------------------------
try:
    transaction = load_data(DATA_PATH)

    st.success("✅ Data loaded successfully!")

    st.subheader("📊 Sample Tourism Data")
    st.dataframe(transaction.head())

    st.subheader("ℹ️ Dataset Info")
    st.write(f"Rows: {transaction.shape[0]}")
    st.write(f"Columns: {transaction.shape[1]}")

except FileNotFoundError:
    st.error("❌ Transaction.xlsx file not found. Please check data folder.")
    st.stop()

# -------------------------------
# Sidebar - User Input
# -------------------------------
st.sidebar.header("🧑 User Input")

visit_year = st.sidebar.number_input(
    "Visit Year",
    min_value=2000,
    max_value=2030,
    value=2022
)

visit_month = st.sidebar.number_input(
    "Visit Month",
    min_value=1,
    max_value=12,
    value=6
)

visit_mode = st.sidebar.selectbox(
    "Visit Mode",
    sorted(transaction["VisitMode"].unique())
)

st.sidebar.markdown("### Selected Input")
st.sidebar.write("Year:", visit_year)
st.sidebar.write("Month:", visit_month)
st.sidebar.write("Mode:", visit_mode)
