import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Earnings Call Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- Data Loading and Caching ---
@st.cache_data
def load_data():
    """Loads, cleans, and prepares the dataset."""
    try:
        df = pd.read_csv('final-prediction.csv')
    except FileNotFoundError:
        st.error("Error: 'final-prediction.csv' not found. Please ensure it's in the same directory.")
        return None

    df['avg_evasiveness'] = pd.to_numeric(df['avg_evasiveness'], errors='coerce')
    median_value = df['avg_evasiveness'].median()
    df['avg_evasiveness'].fillna(median_value, inplace=True)
    df['call_date'] = pd.to_datetime(df['call_date'])
    return df

@st.cache_resource
def train_model(df):
    """Trains models and prepares assets for the app."""
    features = [
        'avg_evasiveness', 'avg_sentiment', 'avg_readability', 'avg_QA_similarity',
        'avg_answer_length', 'avg_numeric_density', 'n_questions'
    ]
    target = 'beat_miss'
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    
    log_reg_model = LogisticRegression(class_weight='balanced', random_state=42).fit(X_train_scaled, y_train)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_scaled, y_train)
    
    return {
        "model": log_reg_model, "scaler": scaler, "features": features,
        "X_test": X_test, "y_test": y_test, "feature_importances": rf_model.feature_importances_
    }

def get_stock_performance(ticker, start_date):
    """
    Fetch stock performance robustly.
    - Handles yfinance v0.2.51+ MultiIndex columns
    - Clamps dates so we don't request future data
    - Handles non-trading days
    - Adds fallback tickers for international listings
    """
    from datetime import datetime

    today = datetime.today().date()

    # If the earnings call is in the future, skip
    if start_date.date() > today:
        st.warning(f"Call date {start_date.date()} is in the future. Skipping stock fetch.")
        return None

    # Possible ticker variants
    possible_tickers = [ticker, f"{ticker}.AS", f"{ticker}.NS", f"{ticker}.L", f"{ticker}.F"]

    # Clamp end date to today (avoid requesting future)
    start_fetch_date = start_date
    end_fetch_date = min(start_date + timedelta(days=120), pd.Timestamp.today())

    stock_data, used_ticker = None, None

    for t in possible_tickers:
        try:
            # FIXED: Set auto_adjust=False to get 'Adj Close' column
            # Set multi_level_index=False to avoid MultiIndex for single ticker
            data = yf.download(
                t, 
                start=start_fetch_date, 
                end=end_fetch_date, 
                auto_adjust=False,
                multi_level_index=False,
                progress=False
            )
            if isinstance(data, pd.DataFrame) and not data.empty:
                stock_data = data.copy()
                used_ticker = t
                break
        except Exception as e:
            continue

    if stock_data is None:
        st.error(f"No data found for {ticker} between {start_fetch_date.date()} and {end_fetch_date.date()}")
        return None

    # FIXED: Simple column name normalization - no MultiIndex flattening needed
    stock_data.columns = [col.lower().strip() for col in stock_data.columns]

    if 'adj close' not in stock_data.columns:
        st.error(f"'Adj Close' column not found. Available columns: {list(stock_data.columns)}")
        return None

    # Find first trading day on/after the call date
    actual_start = stock_data[stock_data.index >= start_date].index.min()
    if pd.isna(actual_start):
        return None

    performance_df = stock_data[stock_data.index >= actual_start].copy()
    if performance_df.empty:
        return None

    # Limit to 60 days (or until today)
    end_60 = min(actual_start + timedelta(days=60), pd.Timestamp.today())
    performance_df = performance_df[performance_df.index <= end_60].copy()
    performance_df['cumulative_return'] = (
        performance_df['adj close'] / performance_df['adj close'].iloc[0]
    ) - 1

    st.caption(f"📊 Data fetched using ticker: **{used_ticker}**, "
               f"from {start_fetch_date.date()} → {end_fetch_date.date()}")

    return performance_df



# --- Main App ---
st.title("📈 Historical Strategy & Analysis Dashboard")
st.markdown("Evaluating a linguistic model's ability to predict earnings surprises and subsequent stock performance.")

df = load_data()

if df is not None:
    model_assets = train_model(df)
    
    # --- Sidebar for Selections ---
    st.sidebar.header("Analysis Selection")
    sorted_tickers = sorted(df['company_ticker'].unique())
    selected_ticker = st.sidebar.selectbox("1. Choose a Company Ticker:", sorted_tickers)
    
    company_df = df[df['company_ticker'] == selected_ticker].sort_values('call_date', ascending=False)
    
    # --- Main Panel ---
    st.header(f"Analysis for: {selected_ticker}")
    
    if not company_df.empty:
        tab1, tab2, tab3 = st.tabs(["**Quarterly Deep Dive**", "**Historical Trends**", "**Model Performance**"])

        with tab1:
            st.subheader("Select a Quarter to Analyze")
            
            date_options = {f"{date.strftime('%Y-%m-%d')} (Q{((date.month-1)//3)+1})": date for date in company_df['call_date']}
            selected_date_str = st.selectbox("2. Select an Earnings Call Date:", options=list(date_options.keys()))
            
            if selected_date_str:
                selected_date = date_options[selected_date_str]
                selected_call = company_df[company_df['call_date'] == selected_date].iloc[0]
                
                features_for_prediction = model_assets['scaler'].transform(selected_call[model_assets['features']].values.reshape(1, -1))
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

                st.subheader("Actual Stock Performance (60 Days Post-Announcement)")
                performance_data = get_stock_performance(selected_ticker, selected_call['call_date'])

                if performance_data is not None:
                    fig_perf = px.line(performance_data, y='cumulative_return',
                                       title=f"Market Reaction after '{actual_text}' on {selected_date_str}",
                                       labels={'cumulative_return': 'Cumulative Return', 'Date': 'Date'})
                    fig_perf.update_yaxes(tickformat=".2%")
                    fig_perf.add_hline(y=0, line_dash="dot", line_color="grey")
                    st.plotly_chart(fig_perf, use_container_width=True)
                else:
                    st.warning(f"Could not retrieve stock performance data for '{selected_ticker}' for this period. The API may have gaps in its historical records.")

        with tab2:
            st.subheader("Historical Linguistic & Financials")
            st.markdown("##### Key Linguistic Metrics Over Time")
            fig_trends = px.line(company_df, x='call_date', y=['avg_evasiveness', 'avg_answer_length', 'avg_numeric_density'],
                                 title=f'Key Linguistic Metrics for {selected_ticker}', markers=True)
            st.plotly_chart(fig_trends, use_container_width=True)
            
            st.markdown("##### Prediction History")
            history_df = company_df.copy()
            history_df['predicted_beat_miss'] = model_assets['model'].predict(model_assets['scaler'].transform(history_df[model_assets['features']]))
            history_df['Actual'] = history_df['beat_miss'].apply(lambda x: 'Beat' if x == 1 else 'Miss')
            history_df['Prediction'] = history_df['predicted_beat_miss'].apply(lambda x: 'Beat' if x == 1 else 'Miss')
            st.dataframe(history_df[['call_date', 'Actual', 'Prediction', 'surprise']].rename(columns={'call_date': 'Date', 'surprise': 'Surprise %'}), use_container_width=True)

        with tab3:
            st.subheader("Overall Model Performance")
            st.markdown("This section shows how our baseline model performs on the entire unseen test set.")
            y_pred = model_assets['model'].predict(model_assets['scaler'].transform(model_assets['X_test']))
            cm = confusion_matrix(model_assets['y_test'], y_pred)
            
            fig_cm, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Miss', 'Beat'], yticklabels=['Miss', 'Beat'], ax=ax)
            ax.set_title('Confusion Matrix (Test Set)')
            ax.set_xlabel('Predicted Outcome')
            ax.set_ylabel('Actual Outcome')
            
            fi_df = pd.DataFrame({
                'feature': model_assets['features'], 'importance': model_assets['feature_importances']
            }).sort_values('importance', ascending=False)
            
            fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=fi_df, ax=ax_fi, palette='viridis')
            ax_fi.set_title('Most Predictive Linguistic Features')
            ax_fi.set_xlabel('Importance Score')
            ax_fi.set_ylabel('Feature')
            
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig_cm)
            with col2:
                st.pyplot(fig_fi)
