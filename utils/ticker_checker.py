import pandas as pd
import yfinance as yf

def check_tickers(filepath='data/final-prediction.csv'):
    print("--- Starting Ticker Analysis ---")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return

    unique_tickers = sorted(df['company_ticker'].unique())
    valid, invalid = [], []

    for ticker in unique_tickers:
        try:
            stock = yf.Ticker(ticker)
            if not stock.history(period="5d").empty:
                print(f"✅ {ticker} is valid")
                valid.append(ticker)
            else:
                print(f"❌ {ticker} invalid")
                invalid.append(ticker)
        except Exception as e:
            print(f"⚠️ Error checking '{ticker}': {e}")
            invalid.append(ticker)

    print(f"\nValid: {len(valid)} | Invalid: {len(invalid)}")
    if invalid:
        print("Invalid tickers:", ", ".join(invalid))

if __name__ == "__main__":
    check_tickers()