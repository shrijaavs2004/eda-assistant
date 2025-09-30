# analysis/app.py

import streamlit as st
import pandas as pd
import os
from google import genai
from dotenv import load_dotenv
from eda_assistant import summarize_data, generate_plot_hist, generate_plot_bar

# Gemini setup
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

st.set_page_config(page_title="EDA Assistant", layout="wide")
st.title("📊 EDA Assistant (AI-powered)")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")
    st.write("### Preview of Data", df.head())

    # -----------------------------
    # Reuse summary from backend
    # -----------------------------
    summary = summarize_data(df)

    st.subheader("📋 Data Summary")
    st.write("**Shape:**", summary["shape"])
    st.write("**Missing Values:**", summary["missing_values"])
    st.write("**Descriptive Statistics:**", summary["description"])

    # -----------------------------
    # Reuse plotting functions
    # -----------------------------
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        st.write("**Histogram of first numeric column**")
        fig = generate_plot_hist(df, numeric_cols[0])
        st.pyplot(fig)

    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        st.write("**Bar Chart of first categorical column**")
        fig = generate_plot_bar(df, cat_cols[0])
        st.pyplot(fig)

    # -----------------------------
    # Gemini Insights
    # -----------------------------
    st.subheader("🤖 AI Insights")
    summary_prompt = f"""
    Dataset columns: {list(df.columns)}
    Missing Values: {summary['missing_values']}
    Dtypes: {summary['dtypes']}

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
