# analysis/eda_assistant.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_csv(file_path):
    return pd.read_csv(file_path)

def summarize_data(df):
    return {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "description": df.describe(include="all").to_dict()
    }

def generate_plot_hist(df, col):
    fig, ax = plt.subplots()
    sns.histplot(df[col].dropna(), kde=True, ax=ax)
    return fig

def generate_plot_bar(df, col):
    fig, ax = plt.subplots()
    df[col].value_counts().plot(kind="bar", ax=ax)
    return fig
