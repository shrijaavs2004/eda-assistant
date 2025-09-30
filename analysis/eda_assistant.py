import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Dataset

def load_csv(file_path="./data/titanic-dataset.csv"):
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    return df

# Basic Summary

def summarize_data(df):
    print("\n📊 Data Summary:")
    print(df.info())
    print("\n🔍 Missing Values:")
    print(df.isnull().sum())
    print("\n📈 Descriptive Statistics:")
    print(df.describe(include="all"))


# Generate Simple Plots

def generate_plots(df, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    
    # Numeric histograms
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols[:2]:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Histogram of {col}")
        plt.savefig(f"{out_dir}/{col}_hist.png")
        plt.close()

    # Categorical bar plots
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols[:2]:
        plt.figure()
        df[col].value_counts().plot(kind='bar')
        plt.title(f"Value Counts of {col}")
        plt.savefig(f"{out_dir}/{col}_bar.png")
        plt.close()

    print(f" Plots saved in '{out_dir}/'")


# Run Everything

if __name__ == "__main__":
    df = load_csv("./data/titanic-dataset.csv")
    summarize_data(df)
    generate_plots(df, out_dir="plots")
