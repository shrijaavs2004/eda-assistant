import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from google import genai
from dotenv import load_dotenv

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

st.set_page_config(page_title="EDA Assistant", layout="wide")
st.title("📊 EDA Assistant (AI-powered)")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")
    st.write("### Preview of Data", df.head())

    # -----------------------------
    # Summary Section
    # -----------------------------
    st.subheader("📋 Data Summary")

    st.write("**Shape:**", df.shape)
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())

    st.write("**Descriptive Statistics:**")
    st.write(df.describe(include="all"))

    # -----------------------------
    # Plots Section
    # -----------------------------
    st.subheader("📈 Auto Plots")

    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        st.write("**Histogram of first numeric column**")
        fig, ax = plt.subplots()
        sns.histplot(df[numeric_cols[0]].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        st.write("**Bar Chart of first categorical column**")
        fig, ax = plt.subplots()
        df[cat_cols[0]].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # -----------------------------
    # Gemini Insights Section
    # -----------------------------
    st.subheader("🤖 AI Insights")

    summary_prompt = f"""
    Dataset columns: {list(df.columns)}
    Missing Values: {df.isnull().sum().to_dict()}
    Dtypes: {df.dtypes.astype(str).to_dict()}

    Write a short analysis of this dataset with:
    - Data quality issues
    - Trends and patterns
    - Next recommended steps
    """

    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=summary_prompt
        )
        st.write(response.text)
    except Exception as e:
        st.error(f"Gemini API error: {e}")
