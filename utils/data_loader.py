import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/final-prediction.csv')
    except FileNotFoundError:
        st.error("File 'data/final-prediction.csv' not found.")
        return None

    df['avg_evasiveness'] = pd.to_numeric(df['avg_evasiveness'], errors='coerce')
    df['avg_evasiveness'].fillna(df['avg_evasiveness'].median(), inplace=True)
    df['call_date'] = pd.to_datetime(df['call_date'])
    return df