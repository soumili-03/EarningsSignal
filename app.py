import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from utils.data_loader import load_data
from models.trainer import train_model
from utils.stock_utils import get_stock_performance_cached
from utils.visuals import plot_confusion_matrix, plot_feature_importance
import time


st.set_page_config(page_title="Earnings Dashboard", page_icon="📈", layout="wide")

st.title("📈 Historical Strategy & Analysis Dashboard")
st.markdown("Evaluating a linguistic model's ability to predict earnings surprises and stock performance.")

df = load_data()
if df is not None:
    model_assets = train_model(df)

    st.sidebar.header("Analysis Selection")
    sorted_tickers = sorted(df['company_ticker'].unique())
    selected_ticker = st.sidebar.selectbox("1. Choose a Company Ticker:", sorted_tickers)

    company_df = df[df['company_ticker'] == selected_ticker].sort_values('call_date', ascending=False)
    st.header(f"Analysis for: {selected_ticker}")

    if not company_df.empty:
        tab1, tab2, tab3 = st.tabs(["Quarterly Deep Dive", "Historical Trends", "Model Performance"])

        # --- Tab 1: Quarterly Analysis ---
        with tab1:
            st.subheader("Select a Quarter to Analyze")
            date_options = {f"{d.strftime('%Y-%m-%d')} (Q{((d.month-1)//3)+1})": d for d in company_df['call_date']}
            selected_date_str = st.selectbox("2. Select Earnings Call Date:", list(date_options.keys()))

            if selected_date_str:
                selected_date = date_options[selected_date_str]
                selected_call = company_df[company_df['call_date'] == selected_date].iloc[0]

                features_for_prediction = model_assets['scaler'].transform(
                    selected_call[model_assets['features']].values.reshape(1, -1)
                )
                prediction = model_assets['model'].predict(features_for_prediction)[0]
                prediction_proba = model_assets['model'].predict_proba(features_for_prediction)[0]

                pred_text = "BEAT ✅" if prediction == 1 else "MISS ⚠️"
                actual_text = "BEAT" if selected_call['beat_miss'] == 1 else "MISS"
                pred_confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]

                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("Model's Prediction", pred_text)
                col2.metric("Actual Outcome", actual_text)
                col3.metric("Model Confidence", f"{pred_confidence:.2%}")

                st.subheader("Stock Performance (60 Days Post-Announcement)")

                try:
                    performance_data = get_stock_performance_cached(selected_ticker, pd.Timestamp(selected_call['call_date']))

                    if performance_data is not None:
                        fig_perf = px.line(performance_data, y='cumulative_return', title=f"Market Reaction after '{actual_text}'")
                        fig_perf.update_yaxes(tickformat=".2%")
                        fig_perf.add_hline(y=0, line_dash="dot", line_color="grey")
                        st.plotly_chart(fig_perf, use_container_width=True)
                    else:
                        st.warning("No stock data found for this period.")
                except Exception as e:
                    st.error(f"Error fetching stock performance: {e}")

        # --- Tab 2: Historical Trends ---
        with tab2:
            st.subheader("Historical Linguistic Trends")
            fig_trends = px.line(company_df, x='call_date', y=['avg_evasiveness', 'avg_answer_length', 'avg_numeric_density'],
                                 title=f'Linguistic Metrics Over Time for {selected_ticker}', markers=True)
            st.plotly_chart(fig_trends, use_container_width=True)

            st.subheader("Prediction History")
            history_df = company_df.copy()
            history_df['predicted'] = model_assets['model'].predict(model_assets['scaler'].transform(history_df[model_assets['features']]))
            history_df['Actual'] = history_df['beat_miss'].map({1: 'Beat', 0: 'Miss'})
            history_df['Predicted'] = history_df['predicted'].map({1: 'Beat', 0: 'Miss'})
            st.dataframe(history_df[['call_date', 'Actual', 'Predicted', 'surprise']].rename(columns={'call_date': 'Date'}))

        # --- Tab 3: Model Performance ---
        with tab3:
            st.subheader("Model Performance Overview")
            y_pred = model_assets['model'].predict(model_assets['scaler'].transform(model_assets['X_test']))
            st.pyplot(plot_confusion_matrix(model_assets['y_test'], y_pred))
            st.pyplot(plot_feature_importance(model_assets['features'], model_assets['feature_importances']))