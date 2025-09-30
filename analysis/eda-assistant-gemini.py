import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# -----------------------------
# 1. Load Dataset
# -----------------------------
def load_csv(file_path="data/sample.csv"):
    df = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    return df

# -----------------------------
# 2. Basic Summary
# -----------------------------
def summarize_data(df):
    summary = {
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "description": df.describe(include="all").to_dict()
    }
    return summary

# -----------------------------
# 3. Generate Simple Plots
# -----------------------------
def generate_plots(df, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols[:2]:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Histogram of {col}")
        plt.savefig(f"{out_dir}/{col}_hist.png")
        plt.close()

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols[:2]:
        plt.figure()
        df[col].value_counts().plot(kind='bar')
        plt.title(f"Value Counts of {col}")
        plt.savefig(f"{out_dir}/{col}_bar.png")
        plt.close()

    print(f"✅ Plots saved in '{out_dir}/'")

# -----------------------------
# 4. Gemini Narrative
# -----------------------------
def generate_gemini_report(summary, model="models/gemini-2.5-flash"):
    client = genai.Client(api_key=API_KEY)
    
    prompt = f"""
    You are a data analyst. Write a short report for this dataset:
    - Missing Values: {summary['missing_values']}
    - Data Types: {summary['dtypes']}
    - Stats (columns): {list(summary['description'].keys())}

    Highlight:
    - Data quality issues
    - Possible trends
    - Next recommended steps
    """

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    return response.text

# -----------------------------
# 5. Run Everything
# -----------------------------
if __name__ == "__main__":
    df = load_csv("data/titanic-dataset.csv")
    summary = summarize_data(df)
    generate_plots(df, out_dir="plots")

    # Generate AI report
    report = generate_gemini_report(summary)
    print("\n📑 Gemini Report:\n", report)

    # Save report
    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.txt", "w") as f:
        f.write(report)
    print("✅ Report saved in 'reports/summary.txt'")
