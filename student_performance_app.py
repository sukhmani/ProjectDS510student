import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data with correct delimiter
df = pd.read_csv("student-mat.csv", sep=";")

# Title
st.title("Predicting Student Performance")

# EDA Section
st.subheader("Exploratory Data Analysis")
st.write("Dataset Overview:")
st.write(df.describe(include='all'))
st.write(f"Shape: {df.shape}")
st.write(f"Columns: {df.columns.tolist()}")

# Correlation Heatmap
st.subheader("Correlation Heatmap")
try:
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    fig, ax = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error generating heatmap: {e}")
