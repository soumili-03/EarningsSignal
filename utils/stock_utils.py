import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import timedelta

@st.cache_data(show_spinner=False)
def get_stock_performance_cached(ticker: str, call_date: pd.Timestamp, days_after: int = 60):
    """
    Fetch stock performance for ~60 trading days after call_date.
    Returns DataFrame with single-level columns: ['close','daily_return','cumulative_return'].
    """
    start_date = call_date
    end_date = call_date + timedelta(days=90)  # buffer to capture ~60 trading days

    try:
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if raw is None or raw.empty:
            st.warning(f"No stock data found for {ticker} between {start_date.date()} and {end_date.date()}.")
            return None

        # --- Flatten MultiIndex columns if present ---
        if isinstance(raw.columns, pd.MultiIndex):
            # join the non-empty parts of the tuple with '_' and normalize
            flat_cols = []
            for col in raw.columns:
                # col is a tuple; filter out empty strings / Nones
                parts = [str(p).strip() for p in col if p not in (None, '')]
                name = "_".join(parts) if parts else str(col)
                flat_cols.append(name)
            raw.columns = flat_cols

        # Normalize column names (lowercase, replace spaces with underscore)
        raw.columns = [str(c).lower().strip().replace(" ", "_") for c in raw.columns]

        # Choose price column: prefer 'adj_close' or 'adjclose', then 'close', then first numeric column
        price_cols_candidates = ['adj_close', 'adjclose', 'close']
        price_col = None
        for c in price_cols_candidates:
            if c in raw.columns:
                price_col = c
                break
        if price_col is None:
            # pick first numeric-like column
            numeric_cols = raw.select_dtypes(include='number').columns
            if len(numeric_cols) > 0:
                price_col = numeric_cols[0]
            else:
                st.error(f"No numeric price column found for {ticker}. Columns: {list(raw.columns)}")
                return None

        # Work on a copy and compute returns
        df = raw.copy()
        df = df.sort_index()

        df['daily_return'] = df[price_col].pct_change()
        # cumulative return from the first available row on/after call_date
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1

        # Produce final DataFrame with canonical column names
        # Ensure 'close' exists as a canonical name (copy price_col -> 'close' if needed)
        if 'close' not in df.columns:
            df['close'] = df[price_col]

        final = df[['close', 'daily_return', 'cumulative_return']].copy()

        # Limit to ~60 trading days (head 60)
        final = final.head(days_after)

        # Ensure index is datetime (Plotly friendly)
        final.index = pd.to_datetime(final.index)

        return final

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None
