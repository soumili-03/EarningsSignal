import os
import re
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from utils.data_loader import load_data
from models.trainer import train_model
from utils.stock_utils import get_stock_performance_cached
from utils.visuals import plot_confusion_matrix, plot_feature_importance

from utils.visuals import plot_classification_report
    

st.set_page_config(page_title="Earnings Dashboard", page_icon="📈", layout="wide")
st.title("📈 Historical Strategy & Analysis Dashboard")
st.markdown("Evaluating a linguistic model's ability to predict earnings surprises and stock performance.")

df = load_data()
if df is not None:
    model_assets = train_model(df)

    st.sidebar.header("Analyze Existing Transcripts")
    sorted_tickers = sorted(df['company_ticker'].unique())

    # --- Tab 1: Read Transcript ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📜 Read Transcript", "Effects on Market", "Get transcript features" ,"Historical trends", "Model Performance"])

    with tab1:
        selected_ticker = st.selectbox("1. Choose a Company Ticker:", sorted_tickers)
        company_df = df[df['company_ticker'] == selected_ticker].sort_values('call_date', ascending=False)

        date_options = {f"{d.strftime('%Y-%m-%d')} (Q{((d.month-1)//3)+1})": d for d in company_df['call_date']}
        selected_date_str = st.selectbox("Select Earnings Call Date:", list(date_options.keys()), key="transcript_date")

        if selected_ticker:
            st.session_state['selected_ticker'] = selected_ticker
        if selected_date_str:
            st.session_state['selected_date'] = date_options[selected_date_str]

        st.header(f"Analysis for: {selected_ticker}")

        if selected_date_str:
            selected_date = date_options[selected_date_str]

            # Build transcript file path
            transcript_dir = "Transcripts"
            ticker_dir = os.path.join(transcript_dir, selected_ticker)
            filename = f"{selected_date.strftime('%Y-%b-%d')}-{selected_ticker}.txt"
            file_path = os.path.join(ticker_dir, filename)

            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    transcript_text = f.read()

                transcript_text = re.sub(r"<Sync[^>]*>", "", transcript_text)
                transcript_text = re.sub(r"</?[^>]+>", "", transcript_text)

                st.markdown(f"### Transcript: {selected_ticker} — {selected_date.strftime('%B %Y')}")
                st.text_area("Earnings Call Transcript", transcript_text, height=500)
            else:
                st.warning(f"No transcript found for {selected_ticker} on {selected_date.strftime('%Y-%m-%d')}.")

    # --- Tab 2: Quarterly Analysis ---
    with tab2:
        if 'selected_ticker' not in st.session_state or 'selected_date' not in st.session_state:
            st.info("Please select a company and quarter in the '📜 Read Transcript' tab to continue.")
        else:
            selected_ticker = st.session_state['selected_ticker']
            selected_date = st.session_state['selected_date']
            company_df = df[df['company_ticker'] == selected_ticker].sort_values('call_date', ascending=False)
            selected_call = company_df[company_df['call_date'] == selected_date].iloc[0]

            features_for_prediction = model_assets['scaler'].transform(
                selected_call[model_assets['features']].values.reshape(1, -1)
            )
            prediction = model_assets['model'].predict(features_for_prediction)[0]
            prediction_proba = model_assets['model'].predict_proba(features_for_prediction)[0]

            pred_text = "BEAT ✅" if prediction == 1 else "MISS ⚠️"
            actual_text = "BEAT" if selected_call['beat_miss'] == 1 else "MISS"
            pred_confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]

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

    # --- Tab 3: Get Transcript Features ---
    with tab3:
        st.header("🧠 Transcript Feature Summary")

        feature_desc_path = os.path.join("data", "feature-descriptions.csv")
        final_pred_path = os.path.join("data", "final-prediction.csv")

        if os.path.exists(feature_desc_path) and os.path.exists(final_pred_path):
            feature_desc_df = pd.read_csv(feature_desc_path)
            final_pred_df = pd.read_csv(final_pred_path)

            # Ensure session state has a selected ticker and date
            if 'selected_ticker' not in st.session_state or 'selected_date' not in st.session_state:
                st.info("Please select a company and quarter in the '📜 Read Transcript' tab to view its feature values.")
            else:
                selected_ticker = st.session_state['selected_ticker']
                selected_date = st.session_state['selected_date']

                # Match the row for the selected call
                selected_call = final_pred_df[
                    (final_pred_df['company_ticker'] == selected_ticker) &
                    (pd.to_datetime(final_pred_df['call_date']) == pd.Timestamp(selected_date))
                ]

                if not selected_call.empty:
                    selected_call = selected_call.iloc[0]
                    feature_values = selected_call.to_dict()

                    # Create {Feature, Value} pairs excluding metadata
                    value_df = pd.DataFrame([
                        {"Feature": k, "Value": v} for k, v in feature_values.items()
                        if k not in ["call_id", "company_ticker", "call_date", "actual_eps", "consensus_eps", "surprise", "beat_miss"]
                    ])

                    merged_df = pd.merge(feature_desc_df, value_df, on="Feature", how="left")
                    merged_df["Value"] = merged_df["Value"].fillna("missing")

                    # Loop through features and display each in a styled card
                    for _, row in merged_df.iterrows():
                        feature_name = row["Feature"].replace("_", " ").title()
                        value = row["Value"]
                        desc = row["Description"]
                        interp = row["Interpretation"]

                        if isinstance(value, (int, float)):
                            value = f"{value:.3f}"

                        st.markdown(f"""
                        <div style="
                            background-color: #f8f9fa;
                            padding: 16px 20px;
                            margin-bottom: 12px;
                            border-radius: 10px;
                            border: 1px solid #e0e0e0;
                            box-shadow: 1px 1px 3px rgba(0,0,0,0.05);
                        ">
                            <h4 style="color:#2c3e50; margin-bottom:4px;">
                                {feature_name}: <span style="color:#0072B2;">{value}</span>
                            </h4>
                            <p style="margin: 6px 0;">
                                <b>Description:</b> {desc}
                            </p>
                            <p style="margin: 6px 0;">
                                <b>Interpretation:</b> {interp}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                else:
                    st.warning(f"No feature data found for {selected_ticker} on {selected_date.strftime('%Y-%m-%d')}.")
        else:
            st.error("Feature or prediction CSV file not found. Please ensure both exist in the 'data/' folder.")


    with tab4:
            
        # ------------------- CHECK SESSION -------------------
        if 'selected_ticker' not in st.session_state or 'selected_date' not in st.session_state:
            st.info("Please select a company and quarter in the '📜 Read Transcript' tab to continue.")
        else:
            selected_ticker = st.session_state['selected_ticker']
            company_df = df[df['company_ticker'] == selected_ticker].sort_values('call_date', ascending=True)

            

            # ------------------- SECTION 2: FEATURE-WISE TRENDS -------------------
            st.markdown("## 🗂️ Feature-wise Trends")

            feature_list = [
                'avg_evasiveness', 'avg_sentiment', 'avg_sentiment_gap',
                'avg_readability_kincaid', 'avg_readability_ease', 'avg_QA_similarity',
                'avg_answer_length', 'avg_numeric_density', 'avg_lm_sentiment',
                'avg_lexical_diversity', 'avg_complex_word_ratio', 'avg_hedge_to_modal_ratio',
                'avg_dale_chall_score', 'avg_sentiment_polarity', 'avg_modal_ratio_verbs',
                'avg_coleman_liau_index', 'avg_filler_freq', 'avg_hedge_freq',
                'avg_passive_rate', 'n_questions'
            ]

            for i in range(0, len(feature_list), 2):
                cols = st.columns(2)
                for j, feature in enumerate(feature_list[i:i+2]):
                    with cols[j]:
                        if feature not in company_df.columns:
                            st.warning(f"Feature '{feature}' missing")
                            continue

                        feature_data = company_df[feature].dropna()
                        if feature_data.empty:
                            st.info(f"No data available for {feature}.")
                            continue

                        # --- Calculate trend change ---
                        if len(feature_data) >= 2:
                            first_val = feature_data.iloc[0]
                            last_val = feature_data.iloc[-1]
                            pct_change = ((last_val - first_val) / first_val) * 100 if first_val != 0 else 0
                            trend_symbol = "🔼" if pct_change > 0 else "🔽"
                            trend_color = "green" if pct_change > 0 else "red"
                            summary = f"{trend_symbol} {abs(pct_change):.2f}% change since first record"
                        else:
                            summary = "No trend data available."
                            trend_color = "gray"

                        # --- Create line chart for feature ---
                        fig = px.line(
                            company_df,
                            x='call_date',
                            y=feature,
                            markers=True,
                            title=feature.replace('_', ' ').title(),
                        )
                        fig.update_traces(line=dict(width=2))
                        fig.update_layout(
                            height=300,
                            template="simple_white",
                            margin=dict(l=10, r=10, t=40, b=10),
                            title_font=dict(size=14, color="#333", family="Arial Black"),
                            xaxis_title="Date",
                            yaxis_title="",
                            plot_bgcolor="rgba(250,250,250,1)",
                            paper_bgcolor="rgba(255,255,255,1)",
                            hovermode="x unified",
                        )

                        # --- Display card ---
                        st.markdown(f"### {feature.replace('_', ' ').title()}")
                        st.markdown(
                            f"<span class='trend-text' style='color:{trend_color};'>{summary}</span>",
                            unsafe_allow_html=True
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # ------------------- SECTION 3: PREDICTION HISTORY -------------------
            st.subheader("📈 Prediction History")

            history_df = company_df.copy()
            history_df['predicted'] = model_assets['model'].predict(
                model_assets['scaler'].transform(history_df[model_assets['features']])
            )
            history_df['Actual'] = history_df['beat_miss'].map({1: 'Beat', 0: 'Miss'})
            history_df['Predicted'] = history_df['predicted'].map({1: 'Beat', 0: 'Miss'})

            st.dataframe(
                history_df[['call_date', 'Actual', 'Predicted', 'surprise']].rename(columns={'call_date': 'Date'}),
                use_container_width=True
            )


    

    with tab5:
        st.subheader("🤖 Overall Model Performance")


        
        y_test = model_assets['y_test']
        y_pred = model_assets['model'].predict(model_assets['scaler'].transform(model_assets['X_test']))
        
        # Overall model stats
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        st.markdown("### Overall Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Accuracy", f"{accuracy:.2%}")
        metric_col2.metric("Precision", f"{precision:.2%}")
        metric_col3.metric("Recall", f"{recall:.2%}")
        metric_col4.metric("F1-Score", f"{f1:.2%}")


        st.pyplot(plot_classification_report(y_test, y_pred))
        
        st.divider()
        
        st.markdown("### Feature Importance")
        st.pyplot(plot_feature_importance(model_assets['features'], model_assets['feature_importances']))
        
        st.divider()
        
        
        