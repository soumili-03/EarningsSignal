import os
import re
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from collections import Counter

from utils.data_loader import load_data
from models.trainer import train_model
from utils.stock_utils import get_stock_performance_cached
from utils.visuals import plot_confusion_matrix, plot_feature_importance
from utils.visuals import plot_classification_report

# Company information dictionary
COMPANY_INFO = {
    'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology', 'color': '#A2AAAD'},
    'AMD': {'name': 'Advanced Micro Devices', 'sector': 'Semiconductors', 'color': '#ED1C24'},
    'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'E-Commerce', 'color': '#FF9900'},
    'ASML': {'name': 'ASML Holding', 'sector': 'Semiconductors', 'color': '#0071C5'},
    'CSCO': {'name': 'Cisco Systems', 'sector': 'Networking', 'color': '#049FD9'},
    'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology', 'color': '#4285F4'},
    'INTC': {'name': 'Intel Corporation', 'sector': 'Semiconductors', 'color': '#0071C5'},
    'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Software', 'color': '#00A4EF'},
    'MU': {'name': 'Micron Technology', 'sector': 'Semiconductors', 'color': '#0A71BA'},
    'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Graphics/AI', 'color': '#76B900'}
}

# Helper function to extract words from transcript
def extract_filler_words(transcript_text):
    """Extract filler words from transcript"""
    filler_words = [
        'um', 'uh', 'hmm', 'er', 'ah', 'like', 'you know', 'sort of', 
        'kind of', 'i mean', 'basically', 'actually', 'literally'
    ]
    
    # Clean and lowercase
    text_lower = transcript_text.lower()
    
    # Find all filler words
    found_fillers = []
    for filler in filler_words:
        count = text_lower.count(filler)
        if count > 0:
            found_fillers.extend([filler] * count)
    
    return found_fillers

def extract_hedge_words(transcript_text):
    """Extract hedge words from transcript"""
    hedge_words = [
        'maybe', 'perhaps', 'possibly', 'probably', 'might', 'could', 
        'would', 'should', 'approximately', 'roughly', 'around', 'about',
        'somewhat', 'fairly', 'relatively', 'generally', 'typically',
        'usually', 'often', 'sometimes', 'occasionally'
    ]
    
    text_lower = transcript_text.lower()
    found_hedges = []
    
    for hedge in hedge_words:
        # Use word boundaries to match whole words only
        pattern = r'\b' + re.escape(hedge) + r'\b'
        matches = re.findall(pattern, text_lower)
        found_hedges.extend(matches)
    
    return found_hedges

def create_word_cloud(words, title="Word Cloud"):
    """Generate word cloud from list of words"""
    if not words or len(words) == 0:
        return None
    
    # Create frequency dictionary
    word_freq = Counter(words)
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        relative_scaling=0.5,
        min_font_size=10
    ).generate_from_frequencies(word_freq)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    return fig

def create_gauge_chart(value, title, min_val=0, max_val=1):
    """Create a gauge chart for a metric"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 14}},
        gauge = {
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'bar': {'color': "#0072B2"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_val, (max_val-min_val)/3 + min_val], 'color': '#ffcccc'},
                {'range': [(max_val-min_val)/3 + min_val, 2*(max_val-min_val)/3 + min_val], 'color': '#ffffcc'},
                {'range': [2*(max_val-min_val)/3 + min_val, max_val], 'color': '#ccffcc'}
            ],
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'size': 12}
    )
    
    return fig

def create_bar_comparison(value, feature_name, df, selected_ticker):
    """Create a bar chart comparing selected call to company average"""
    company_df = df[df['company_ticker'] == selected_ticker]
    company_avg = company_df[feature_name].mean()
    overall_avg = df[feature_name].mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['This Call', 'Company Avg', 'Overall Avg'],
        y=[value, company_avg, overall_avg],
        marker_color=['#0072B2', '#56B4E9', '#999999'],
        text=[f'{value:.3f}', f'{company_avg:.3f}', f'{overall_avg:.3f}'],
        textposition='auto',
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240,240,240,0.5)",
        yaxis_title="Value",
        showlegend=False
    )
    
    return fig

def create_company_mini_chart(df, ticker):
    company_data = df[df['company_ticker'] == ticker].sort_values('call_date')
    if len(company_data) < 2:
        return None, 0
    beat_rate = (company_data['beat_miss'].sum() / len(company_data)) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=company_data['call_date'],y=company_data['surprise'] * 100, mode='lines+markers', line=dict(color=COMPANY_INFO[ticker]['color'], width=2), marker=dict(size=6), showlegend=False, hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Surprise: %{y:.1f}%<extra></extra>'))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(height=150, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.02)", xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False))
    return fig, beat_rate
    
st.set_page_config(page_title="Earnings Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
# Custom CSS
st.markdown("""<style>.main-header {text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 2rem;} .metric-card {background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 1.5rem; border-radius: 10px; text-align: center; margin: 0.5rem 0;} .big-number {font-size: 2.5rem; font-weight: bold; color: #2c3e50;} .label {font-size: 0.9rem; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px;}</style>""", unsafe_allow_html=True)

df = load_data()
if df is not None:
    model_assets = train_model(df)
    total_calls = len(df)
    total_companies = df['company_ticker'].nunique()
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Get metrics for the best model
    y_test = model_assets['y_test']
    y_pred = model_assets['model'].predict(model_assets['scaler'].transform(model_assets['X_test']))
    model_accuracy = accuracy_score(y_test, y_pred)
    model_precision = precision_score(y_test, y_pred, zero_division=0)
    model_recall = recall_score(y_test, y_pred, zero_division=0)
    model_f1 = f1_score(y_test, y_pred, zero_division=0)

    # Get predictions for both models for comparison
    lr_pred = model_assets['lr_model'].predict(model_assets['lr_scaler'].transform(model_assets['lr_X_test']))
    rf_pred = model_assets['rf_model'].predict(model_assets['rf_scaler'].transform(model_assets['rf_X_test']))

    st.sidebar.header("Analyze Existing Transcripts")
    sorted_tickers = sorted(df['company_ticker'].unique())


    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏠 Dashboard","📜 Read Transcript", "📊Effects on Market", "🧠 Get transcript features" ,"📈 Historical trends", "Model Comparison"])
    
    #tab 1: dashboard
    with tab1:
        st.markdown('<div class="main-header"><h1>🤖 Earnings Call Analytics</h1><p style="font-size: 1.2rem; margin-top: 0.5rem;">Advanced NLP-powered predictions for earnings surprises</p></div>', unsafe_allow_html=True)
        
        st.markdown("## 📊 Platform Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="big-number">{total_calls}</div><div class="label">Earnings Calls</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="big-number">{total_companies}</div><div class="label">Companies</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="big-number">{model_assets["best_model_name"]}</div><div class="label">Best Model</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="big-number">{model_accuracy*100:.1f}%</div><div class="label">Accuracy</div></div>', unsafe_allow_html=True)
        with col5:
            st.markdown(f'<div class="metric-card"><div class="big-number">{model_f1*100:.1f}%</div><div class="label">F1 Score</div></div>', unsafe_allow_html=True)
        
        st.divider()
        st.markdown("## 🎯 Model Performance")
        
        perf_col1, perf_col2 = st.columns([1, 1])
        with perf_col1:
            st.markdown("### Classification Metrics")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Accuracy", f"{model_accuracy*100:.2f}%", delta="High Confidence")
                st.metric("F1-Score", f"{model_f1*100:.2f}%", delta="Balanced")
            with m2:
                st.metric("Precision", f"{model_precision*100:.2f}%", delta="Low False Positives")
                st.metric("Recall", f"{model_recall*100:.2f}%", delta="Catches True Beats")
            st.pyplot(plot_classification_report(y_test, y_pred))
        
        with perf_col2:
            st.markdown("### Feature Importance")
            st.caption("Top linguistic features that predict earnings surprises")
            st.pyplot(plot_feature_importance(model_assets['features'], model_assets['feature_importances']))
        
        st.divider()
        st.markdown("## 🏢 Companies We Analyze")
        st.caption("Click on any company to explore their earnings call history")
        
        tickers = sorted(df['company_ticker'].unique())
        cols_per_row = 2
        for i in range(0, len(tickers), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, ticker in enumerate(tickers[i:i+cols_per_row]):
                with cols[j]:
                    company_name = COMPANY_INFO.get(ticker, {}).get('name', ticker)
                    sector = COMPANY_INFO.get(ticker, {}).get('sector', 'Tech')
                    company_data = df[df['company_ticker'] == ticker].sort_values('call_date')
                    num_calls = len(company_data)
                    beat_rate = (company_data['beat_miss'].sum() / len(company_data)) * 100
                    avg_surprise = company_data['surprise'].mean() * 100
                    
                    with st.expander(f"**{ticker}** - {company_name}", expanded=False):
                        st.markdown(f"**Sector:** {sector}")
                        st.markdown(f"**Calls Analyzed:** {num_calls}")
                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("Beat Rate", f"{beat_rate:.1f}%", delta=f"{beat_rate-50:.1f}% vs 50%")
                        with m2:
                            st.metric("Avg Surprise", f"{avg_surprise:+.2f}%", delta="Average" if abs(avg_surprise) < 5 else "Significant")
                        chart_data, _ = create_company_mini_chart(df, ticker)
                        if chart_data:
                            st.plotly_chart(chart_data, use_container_width=True)
                        recent = company_data.tail(3)[['call_date', 'beat_miss', 'surprise']]
                        recent['Result'] = recent['beat_miss'].map({1: '✅ Beat', 0: '❌ Miss'})
                        recent['Surprise'] = (recent['surprise'] * 100).apply(lambda x: f"{x:+.2f}%")
                        recent['Date'] = pd.to_datetime(recent['call_date']).dt.strftime('%Y-%m-%d')
                        st.markdown("**Recent Calls:**")
                        st.dataframe(recent[['Date', 'Result', 'Surprise']].sort_values('Date', ascending=False), use_container_width=True, hide_index=True)
        
        st.divider()
        st.markdown("## 🔮 Recent Predictions")
        st.caption("Latest earnings calls analyzed by our model")
        
        recent_calls = df.sort_values('call_date', ascending=False).head(5).copy()
        recent_calls['predicted'] = model_assets['model'].predict(model_assets['scaler'].transform(recent_calls[model_assets['features']]))
        recent_calls['confidence'] = model_assets['model'].predict_proba(model_assets['scaler'].transform(recent_calls[model_assets['features']])).max(axis=1)
        
        for idx, row in recent_calls.iterrows():
            col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 2, 2])
            with col1:
                st.markdown(f"**{row['company_ticker']}**")
                st.caption(COMPANY_INFO.get(row['company_ticker'], {}).get('name', row['company_ticker']))
            with col2:
                st.caption("Date")
                st.write(pd.to_datetime(row['call_date']).strftime('%Y-%m-%d'))
            with col3:
                actual = "✅ Beat" if row['beat_miss'] == 1 else "❌ Miss"
                st.caption("Actual")
                st.write(actual)
            with col4:
                predicted = "✅ Beat" if row['predicted'] == 1 else "❌ Miss"
                correct = "✅" if row['beat_miss'] == row['predicted'] else "❌"
                st.caption("Predicted")
                st.write(f"{predicted} {correct}")
            with col5:
                st.caption("Confidence")
                st.write(f"{row['confidence']*100:.1f}%")
        
        st.divider()
        st.markdown("## 📈 Performance by Company")
        
        company_stats = []
        for ticker in tickers:
            ticker_data = df[df['company_ticker'] == ticker].copy()
            ticker_data['predicted'] = model_assets['model'].predict(model_assets['scaler'].transform(ticker_data[model_assets['features']]))
            accuracy = (ticker_data['beat_miss'] == ticker_data['predicted']).mean()
            beat_rate = ticker_data['beat_miss'].mean()
            num_calls = len(ticker_data)
            company_stats.append({'Company': ticker, 'Name': COMPANY_INFO.get(ticker, {}).get('name', ticker), 'Calls': num_calls, 'Beat Rate': f"{beat_rate*100:.1f}%", 'Model Accuracy': f"{accuracy*100:.1f}%"})
        
        stats_df = pd.DataFrame(company_stats).sort_values('Model Accuracy', ascending=False)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        fig_accuracy = go.Figure()
        accuracy_values = [float(x.strip('%')) for x in stats_df['Model Accuracy']]
        colors = ['#2ecc71' if x >= 70 else '#f39c12' if x >= 60 else '#e74c3c' for x in accuracy_values]
        fig_accuracy.add_trace(go.Bar(x=stats_df['Company'], y=accuracy_values, marker_color=colors, text=[f"{x:.1f}%" for x in accuracy_values], textposition='outside', hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.1f}%<extra></extra>'))
        fig_accuracy.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Good Performance (70%)", annotation_position="right")
        fig_accuracy.update_layout(title="Model Accuracy by Company", xaxis_title="Company", yaxis_title="Accuracy (%)", height=400, template="plotly_white", showlegend=False)
        st.plotly_chart(fig_accuracy, use_container_width=True)
    
    # --- Tab 2: Read Transcript ---
    with tab2:
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
                
                # Store transcript in session state for use in other tabs
                st.session_state['current_transcript'] = transcript_text
            else:
                st.warning(f"No transcript found for {selected_ticker} on {selected_date.strftime('%Y-%m-%d')}.")
                st.session_state['current_transcript'] = None

    # --- Tab 3: ENHANCED Effects on Market ---
    with tab3:
        if 'selected_ticker' not in st.session_state or 'selected_date' not in st.session_state:
            st.info("Please select a company and quarter in the '📜 Read Transcript' tab to continue.")
        else:
            selected_ticker = st.session_state['selected_ticker']
            selected_date = st.session_state['selected_date']
            company_df = df[df['company_ticker'] == selected_ticker].sort_values('call_date', ascending=False)
            selected_call = company_df[company_df['call_date'] == selected_date].iloc[0]

            # Get model prediction
            features_for_prediction = model_assets['scaler'].transform(
                selected_call[model_assets['features']].values.reshape(1, -1)
            )
            prediction = model_assets['model'].predict(features_for_prediction)[0]
            prediction_proba = model_assets['model'].predict_proba(features_for_prediction)[0]

            pred_text = "BEAT ✅" if prediction == 1 else "MISS ⚠️"
            actual_text = "BEAT" if selected_call['beat_miss'] == 1 else "MISS"
            pred_confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
            
            # Get actual earnings data
            actual_eps = selected_call['actual_eps']
            consensus_eps = selected_call['consensus_eps']
            surprise = selected_call['surprise']
            
            st.header("📊 Market Impact Analysis")
            st.markdown(f"### {selected_ticker} — {selected_date.strftime('%B %d, %Y')}")
            
            # --- SECTION 1: Key Metrics Overview ---
            st.markdown("#### 🎯 Earnings Call Summary")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Actual Result",
                    actual_text,
                    delta=None,
                    delta_color="normal"
                )
            
            with col2:
                surprise_color = "normal" if surprise >= 0 else "inverse"
                st.metric(
                    "Surprise %",
                    f"{surprise*100:.2f}%",
                    delta=f"{'Beat' if surprise > 0 else 'Miss'} by {abs(surprise*100):.2f}%",
                    delta_color=surprise_color
                )
            
            with col3:
                st.metric(
                    "Actual EPS",
                    f"${actual_eps:.2f}",
                    delta=None
                )
            
            with col4:
                st.metric(
                    "Consensus EPS",
                    f"${consensus_eps:.2f}",
                    delta=None
                )
            
            with col5:
                st.metric(
                    "Model Prediction",
                    pred_text,
                    delta=f"{pred_confidence*100:.1f}% confident",
                    delta_color="off"
                )
            
            # Model accuracy indicator
            if (prediction == 1 and selected_call['beat_miss'] == 1) or (prediction == 0 and selected_call['beat_miss'] == 0):
                st.success("✅ **Model Prediction: CORRECT**")
            else:
                st.error("❌ **Model Prediction: INCORRECT**")
            
            st.divider()
            
            # --- SECTION 2: Stock Performance Analysis ---
            try:
                performance_data = get_stock_performance_cached(selected_ticker, pd.Timestamp(selected_call['call_date']))

                if performance_data is not None:
                    st.markdown("#### 📈 Stock Price Movement (60 Trading Days Post-Call)")
                    
                    # Calculate key metrics from performance data
                    final_return = performance_data['cumulative_return'].iloc[-1]
                    max_return = performance_data['cumulative_return'].max()
                    min_return = performance_data['cumulative_return'].min()
                    volatility = performance_data['daily_return'].std()
                    
                    # Display performance metrics
                    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                    
                    with perf_col1:
                        return_color = "normal" if final_return >= 0 else "inverse"
                        st.metric(
                            "60-Day Return",
                            f"{final_return*100:.2f}%",
                            delta=f"{'Gain' if final_return >= 0 else 'Loss'}",
                            delta_color=return_color
                        )
                    
                    with perf_col2:
                        st.metric(
                            "Peak Return",
                            f"{max_return*100:.2f}%",
                            delta="Highest point"
                        )
                    
                    with perf_col3:
                        st.metric(
                            "Lowest Return",
                            f"{min_return*100:.2f}%",
                            delta="Lowest point",
                            delta_color="inverse"
                        )
                    
                    with perf_col4:
                        st.metric(
                            "Volatility (σ)",
                            f"{volatility*100:.2f}%",
                            delta="Daily std dev",
                            delta_color="off"
                        )
                    
                    # Create enhanced cumulative return chart
                    fig_perf = go.Figure()
                    
                    # Add cumulative return line
                    fig_perf.add_trace(go.Scatter(
                        x=performance_data.index,
                        y=performance_data['cumulative_return'] * 100,
                        mode='lines',
                        name='Cumulative Return',
                        line=dict(color='#0072B2', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(0, 114, 178, 0.1)'
                    ))
                    
                    # Add zero line
                    fig_perf.add_hline(
                        y=0, 
                        line_dash="dash", 
                        line_color="gray",
                        annotation_text="Break Even",
                        annotation_position="right"
                    )
                    
                    # Add positive/negative zones
                    if max_return > 0:
                        fig_perf.add_hrect(
                            y0=0, y1=max_return*100,
                            fillcolor="green", opacity=0.05,
                            layer="below", line_width=0
                        )
                    if min_return < 0:
                        fig_perf.add_hrect(
                            y0=min_return*100, y1=0,
                            fillcolor="red", opacity=0.05,
                            layer="below", line_width=0
                        )
                    
                    fig_perf.update_layout(
                        title=f"Cumulative Stock Return After {actual_text} Announcement",
                        xaxis_title="Trading Days Since Call",
                        yaxis_title="Cumulative Return (%)",
                        height=450,
                        hovermode='x unified',
                        template="plotly_white",
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    
                    st.plotly_chart(fig_perf, use_container_width=True)
                    
                    st.divider()
                    
                    # --- SECTION 3: Detailed Analysis ---
                    analysis_col1, analysis_col2 = st.columns(2)
                    
                    with analysis_col1:
                        st.markdown("#### 📊 Daily Returns Distribution")
                        
                        # Create histogram of daily returns
                        fig_hist = go.Figure()
                        
                        fig_hist.add_trace(go.Histogram(
                            x=performance_data['daily_return'] * 100,
                            nbinsx=30,
                            name='Daily Returns',
                            marker_color='#56B4E9',
                            opacity=0.75
                        ))
                        
                        fig_hist.update_layout(
                            xaxis_title="Daily Return (%)",
                            yaxis_title="Frequency",
                            height=300,
                            template="plotly_white",
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Stats
                        st.markdown("**Key Statistics:**")
                        st.write(f"• Mean Daily Return: {performance_data['daily_return'].mean()*100:.3f}%")
                        st.write(f"• Median Daily Return: {performance_data['daily_return'].median()*100:.3f}%")
                        st.write(f"• Positive Days: {(performance_data['daily_return'] > 0).sum()} / {len(performance_data)}")
                    
                    with analysis_col2:
                        st.markdown("#### 📅 Return Milestones")
                        
                        # Create milestones
                        milestones = []
                        
                        # First day return
                        if len(performance_data) > 0:
                            day1_return = performance_data['cumulative_return'].iloc[0]
                            milestones.append(("Day 1", day1_return))
                        
                        # First week return (5 trading days)
                        if len(performance_data) >= 5:
                            week1_return = performance_data['cumulative_return'].iloc[4]
                            milestones.append(("Week 1", week1_return))
                        
                        # First month return (~20 trading days)
                        if len(performance_data) >= 20:
                            month1_return = performance_data['cumulative_return'].iloc[19]
                            milestones.append(("Month 1", month1_return))
                        
                        # Final return
                        if len(performance_data) > 0:
                            final_return = performance_data['cumulative_return'].iloc[-1]
                            milestones.append(("60 Days", final_return))
                        
                        # Create milestone chart
                        if milestones:
                            milestone_names = [m[0] for m in milestones]
                            milestone_values = [m[1] * 100 for m in milestones]
                            
                            colors = ['green' if v >= 0 else 'red' for v in milestone_values]
                            
                            fig_milestone = go.Figure()
                            
                            fig_milestone.add_trace(go.Bar(
                                x=milestone_names,
                                y=milestone_values,
                                marker_color=colors,
                                text=[f"{v:.2f}%" for v in milestone_values],
                                textposition='outside'
                            ))
                            
                            fig_milestone.update_layout(
                                yaxis_title="Cumulative Return (%)",
                                height=300,
                                template="plotly_white",
                                showlegend=False
                            )
                            
                            fig_milestone.add_hline(y=0, line_dash="dash", line_color="gray")
                            
                            st.plotly_chart(fig_milestone, use_container_width=True)
                    
                    st.divider()
                    
                    # --- SECTION 4: Interpretation ---
                    st.markdown("#### 🧠 Market Reaction Interpretation")
                    
                    # Determine market reaction
                    if final_return > 0.05:  # >5% gain
                        reaction = "**Strong Positive** 🚀"
                        reaction_color = "green"
                        explanation = "The market reacted very positively to this earnings call, with the stock gaining over 5% in the following 60 trading days."
                    elif final_return > 0:  # 0-5% gain
                        reaction = "**Moderately Positive** 📈"
                        reaction_color = "green"
                        explanation = "The market showed a positive but modest reaction to this earnings call."
                    elif final_return > -0.05:  # 0 to -5% loss
                        reaction = "**Slightly Negative** 📉"
                        reaction_color = "orange"
                        explanation = "The market reacted with a slight decline following this earnings call."
                    else:  # < -5% loss
                        reaction = "**Strongly Negative** ⚠️"
                        reaction_color = "red"
                        explanation = "The market reacted negatively to this earnings call, with the stock declining more than 5% over the next 60 trading days."
                    
                    st.markdown(f"**Overall Market Reaction:** {reaction}")
                    st.info(explanation)
                    
                    # Alignment analysis
                    earnings_positive = surprise > 0
                    market_positive = final_return > 0
                    
                    if earnings_positive and market_positive:
                        st.success("✅ **Aligned:** The positive earnings surprise translated into positive market performance.")
                    elif not earnings_positive and not market_positive:
                        st.warning("⚠️ **Aligned:** The earnings miss was followed by negative market performance.")
                    elif earnings_positive and not market_positive:
                        st.error("❌ **Misaligned:** Despite beating estimates, the stock declined. This could indicate concerns about guidance, management commentary, or broader market conditions.")
                    else:
                        st.info("📊 **Misaligned:** Despite missing estimates, the stock gained value. This could suggest the market had lower expectations or found positive signals in the call.")
                    
                else:
                    st.warning("No stock data found for this period.")
            except Exception as e:
                st.error(f"Error fetching stock performance: {e}")

    # --- Tab 4: ENHANCED Get Transcript Features ---
    with tab4:
        st.header("🧠 Transcript Feature Analysis")

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
                    
                    # Create feature categories
                    sentiment_features = ['avg_sentiment', 'avg_sentiment_gap', 'avg_lm_sentiment', 'avg_sentiment_polarity']
                    readability_features = ['avg_readability_kincaid', 'avg_readability_ease', 'avg_dale_chall_score', 'avg_coleman_liau_index']
                    linguistic_features = ['avg_evasiveness', 'avg_lexical_diversity', 'avg_complex_word_ratio', 'avg_filler_freq', 'avg_hedge_freq', 'avg_hedge_to_modal_ratio', 'avg_modal_ratio_verbs', 'avg_passive_rate']
                    qa_features = ['avg_QA_similarity', 'avg_answer_length', 'n_questions', 'avg_numeric_density']
                    
                    # Category tabs
                    cat_tab1, cat_tab2, cat_tab3, cat_tab4 = st.tabs([
                        "💬 Sentiment Analysis",
                        "📖 Readability Metrics",
                        "🗣️ Linguistic Patterns",
                        "❓ Q&A Dynamics"
                    ])
                    
                    # --- SENTIMENT TAB ---
                    with cat_tab1:
                        st.subheader("Sentiment Indicators")
                        
                        col1, col2 = st.columns(2)
                        
                        for i, feature in enumerate(sentiment_features):
                            if feature in selected_call:
                                value = selected_call[feature]
                                
                                # Get description
                                desc_row = feature_desc_df[feature_desc_df['Feature'] == feature]
                                desc = desc_row['Description'].values[0] if not desc_row.empty else "No description"
                                interp = desc_row['Interpretation'].values[0] if not desc_row.empty else "No interpretation"
                                
                                with (col1 if i % 2 == 0 else col2):
                                    with st.expander(f"📊 {feature.replace('avg_', '').replace('_', ' ').title()}", expanded=False):
                                        st.markdown(f"**Value:** `{value:.4f}`")
                                        st.caption(desc)
                                        
                                        # Gauge chart for sentiment
                                        if 'sentiment' in feature:
                                            st.plotly_chart(create_gauge_chart(
                                                value, 
                                                feature.replace('avg_', '').replace('_', ' ').title(),
                                                min_val=-1 if value < 0 else 0,
                                                max_val=1
                                            ), use_container_width=True)
                                        
                                        # Comparison bar
                                        st.plotly_chart(create_bar_comparison(value, feature, df, selected_ticker), use_container_width=True)
                                        
                                        st.info(f"💡 {interp}")
                    
                    # --- READABILITY TAB ---
                    with cat_tab2:
                        st.subheader("Readability & Complexity Scores")
                        
                        col1, col2 = st.columns(2)
                        
                        for i, feature in enumerate(readability_features):
                            if feature in selected_call:
                                value = selected_call[feature]
                                
                                desc_row = feature_desc_df[feature_desc_df['Feature'] == feature]
                                desc = desc_row['Description'].values[0] if not desc_row.empty else "No description"
                                interp = desc_row['Interpretation'].values[0] if not desc_row.empty else "No interpretation"
                                
                                with (col1 if i % 2 == 0 else col2):
                                    with st.expander(f"📚 {feature.replace('avg_', '').replace('_', ' ').title()}", expanded=False):
                                        st.markdown(f"**Value:** `{value:.4f}`")
                                        st.caption(desc)
                                        
                                        # Gauge for readability
                                        st.plotly_chart(create_gauge_chart(
                                            value, 
                                            feature.replace('avg_', '').replace('_', ' ').title(),
                                            min_val=0,
                                            max_val=max(100, value * 1.2)
                                        ), use_container_width=True)
                                        
                                        st.plotly_chart(create_bar_comparison(value, feature, df, selected_ticker), use_container_width=True)
                                        
                                        st.info(f"💡 {interp}")
                    
                    # --- LINGUISTIC TAB ---
                    with cat_tab3:
                        st.subheader("Linguistic Patterns & Communication Style")
                        
                        # Special handling for filler and hedge words
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🎤 Filler Words Analysis")
                            
                            filler_value = selected_call.get('avg_filler_freq', 0)
                            st.metric("Filler Word Frequency", f"{filler_value:.4f}")
                            
                            # Generate word cloud if transcript available
                            if 'current_transcript' in st.session_state and st.session_state['current_transcript']:
                                filler_words = extract_filler_words(st.session_state['current_transcript'])
                                
                                if filler_words:
                                    st.caption(f"Found {len(filler_words)} filler words in transcript")
                                    fig_wc = create_word_cloud(filler_words, "Filler Words Used")
                                    if fig_wc:
                                        st.pyplot(fig_wc)
                                        
                                        # Top filler words
                                        filler_counts = Counter(filler_words)
                                        st.markdown("**Most Common Fillers:**")
                                        for word, count in filler_counts.most_common(5):
                                            st.write(f"• **{word}**: {count} times")
                                else:
                                    st.info("No filler words detected in this transcript")
                            else:
                                st.warning("Load transcript in 'Read Transcript' tab to see word cloud")
                        
                        with col2:
                            st.markdown("### 🤔 Hedge Words Analysis")
                            
                            hedge_value = selected_call.get('avg_hedge_freq', 0)
                            st.metric("Hedge Word Frequency", f"{hedge_value:.4f}")
                            
                            # Generate word cloud for hedge words
                            if 'current_transcript' in st.session_state and st.session_state['current_transcript']:
                                hedge_words = extract_hedge_words(st.session_state['current_transcript'])
                                
                                if hedge_words:
                                    st.caption(f"Found {len(hedge_words)} hedge words in transcript")
                                    fig_wc = create_word_cloud(hedge_words, "Hedge Words Used")
                                    if fig_wc:
                                        st.pyplot(fig_wc)
                                        
                                        # Top hedge words
                                        hedge_counts = Counter(hedge_words)
                                        st.markdown("**Most Common Hedges:**")
                                        for word, count in hedge_counts.most_common(5):
                                            st.write(f"• **{word}**: {count} times")
                                else:
                                    st.info("No hedge words detected in this transcript")
                            else:
                                st.warning("Load transcript in 'Read Transcript' tab to see word cloud")
                        
                        st.markdown("---")
                        
                        # Other linguistic features
                        st.markdown("### 📝 Additional Linguistic Features")
                        col3, col4 = st.columns(2)
                        
                        other_features = [f for f in linguistic_features if f not in ['avg_filler_freq', 'avg_hedge_freq']]
                        
                        for i, feature in enumerate(other_features):
                            if feature in selected_call:
                                value = selected_call[feature]
                                
                                desc_row = feature_desc_df[feature_desc_df['Feature'] == feature]
                                desc = desc_row['Description'].values[0] if not desc_row.empty else "No description"
                                interp = desc_row['Interpretation'].values[0] if not desc_row.empty else "No interpretation"
                                
                                with (col3 if i % 2 == 0 else col4):
                                    with st.expander(f"📝 {feature.replace('avg_', '').replace('_', ' ').title()}", expanded=False):
                                        st.markdown(f"**Value:** `{value:.4f}`")
                                        st.caption(desc)
                                        
                                        st.plotly_chart(create_bar_comparison(value, feature, df, selected_ticker), use_container_width=True)
                                        
                                        st.info(f"💡 {interp}")
                    
                    # --- Q&A TAB ---
                    with cat_tab4:
                        st.subheader("Q&A Session Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        for i, feature in enumerate(qa_features):
                            if feature in selected_call:
                                value = selected_call[feature]
                                
                                desc_row = feature_desc_df[feature_desc_df['Feature'] == feature]
                                desc = desc_row['Description'].values[0] if not desc_row.empty else "No description"
                                interp = desc_row['Interpretation'].values[0] if not desc_row.empty else "No interpretation"
                                
                                with (col1 if i % 2 == 0 else col2):
                                    with st.expander(f"❓ {feature.replace('avg_', '').replace('_', ' ').title()}", expanded=False):
                                        st.markdown(f"**Value:** `{value:.4f}` {'' if feature != 'n_questions' else 'questions'}")
                                        st.caption(desc)
                                        
                                        if feature != 'n_questions':
                                            st.plotly_chart(create_bar_comparison(value, feature, df, selected_ticker), use_container_width=True)
                                        else:
                                            # Bar chart for number of questions
                                            company_avg_q = df[df['company_ticker'] == selected_ticker]['n_questions'].mean()
                                            overall_avg_q = df['n_questions'].mean()
                                            
                                            fig = go.Figure()
                                            fig.add_trace(go.Bar(
                                                x=['This Call', 'Company Avg', 'Overall Avg'],
                                                y=[value, company_avg_q, overall_avg_q],
                                                marker_color=['#0072B2', '#56B4E9', '#999999'],
                                                text=[f'{int(value)}', f'{company_avg_q:.0f}', f'{overall_avg_q:.0f}'],
                                                textposition='auto',
                                            ))
                                            fig.update_layout(
                                                height=250,
                                                margin=dict(l=20, r=20, t=20, b=40),
                                                paper_bgcolor="rgba(0,0,0,0)",
                                                plot_bgcolor="rgba(240,240,240,0.5)",
                                                yaxis_title="Number of Questions",
                                                showlegend=False
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                        
                                        st.info(f"💡 {interp}")

                else:
                    st.warning(f"No feature data found for {selected_ticker} on {selected_date.strftime('%Y-%m-%d')}.")
        else:
            st.error("Feature or prediction CSV file not found. Please ensure both exist in the 'data/' folder.")


    with tab5:
            
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

    with tab6:
        st.markdown("## 🤖 Model Comparison: Logistic Regression vs Random Forest")
        
        st.info(f"**Selected Model:** {model_assets['best_model_name']} (based on highest F1 score)")
        
        # Metrics Comparison Table
        st.markdown("### 📊 Performance Metrics Comparison")
        
        comparison_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Logistic Regression': [
                f"{model_assets['lr_metrics']['accuracy']*100:.2f}%",
                f"{model_assets['lr_metrics']['precision']*100:.2f}%",
                f"{model_assets['lr_metrics']['recall']*100:.2f}%",
                f"{model_assets['lr_metrics']['f1']*100:.2f}%"
            ],
            'Random Forest': [
                f"{model_assets['rf_metrics']['accuracy']*100:.2f}%",
                f"{model_assets['rf_metrics']['precision']*100:.2f}%",
                f"{model_assets['rf_metrics']['recall']*100:.2f}%",
                f"{model_assets['rf_metrics']['f1']*100:.2f}%"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Visual comparison
        st.markdown("### 📈 Visual Performance Comparison")
        
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        lr_values = [
            model_assets['lr_metrics']['accuracy'] * 100,
            model_assets['lr_metrics']['precision'] * 100,
            model_assets['lr_metrics']['recall'] * 100,
            model_assets['lr_metrics']['f1'] * 100
        ]
        rf_values = [
            model_assets['rf_metrics']['accuracy'] * 100,
            model_assets['rf_metrics']['precision'] * 100,
            model_assets['rf_metrics']['recall'] * 100,
            model_assets['rf_metrics']['f1'] * 100
        ]
        
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            name='Logistic Regression',
            x=metrics_names,
            y=lr_values,
            marker_color='#0072B2',
            text=[f'{v:.2f}%' for v in lr_values],
            textposition='outside'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Random Forest',
            x=metrics_names,
            y=rf_values,
            marker_color='#D55E00',
            text=[f'{v:.2f}%' for v in rf_values],
            textposition='outside'
        ))
        
        fig_comparison.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Metrics",
            yaxis_title="Score (%)",
            barmode='group',
            height=400,
            template="plotly_white",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Confusion matrices side by side
        st.markdown("### 🎯 Confusion Matrices")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Logistic Regression")
            st.pyplot(plot_confusion_matrix(model_assets['lr_y_test'], lr_pred))
        
        with col2:
            st.markdown("#### Random Forest")
            st.pyplot(plot_confusion_matrix(model_assets['rf_y_test'], rf_pred))
        
        # Classification reports
        st.markdown("### 📋 Detailed Classification Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Logistic Regression")
            st.pyplot(plot_classification_report(model_assets['lr_y_test'], lr_pred))
        
        with col2:
            st.markdown("#### Random Forest")
            st.pyplot(plot_classification_report(model_assets['rf_y_test'], rf_pred))
        
        # Model insights
        st.markdown("### 💡 Model Insights")
        
        if model_assets['best_model_name'] == "Logistic Regression":
            st.success("""
            **Why Logistic Regression Won:**
            - Better generalization on validation data
            - Lower risk of overfitting
            - More interpretable predictions
            - Faster training and inference
            """)
        else:
            st.success("""
            **Why Random Forest Won:**
            - Better captures non-linear relationships
            - Handles feature interactions effectively
            - More robust to outliers
            - Ensemble approach reduces variance
            """)