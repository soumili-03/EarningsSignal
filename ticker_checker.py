import pandas as pd
import yfinance as yf

def check_tickers(filepath='final-prediction.csv'):
    """
    Loads tickers from the CSV and checks their validity against the yfinance API.
    """
    print("--- Starting Ticker Analysis ---")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return

    unique_tickers = sorted(df['company_ticker'].unique())
    print(f"Found {len(unique_tickers)} unique tickers to test.\n")

    valid_tickers = []
    invalid_tickers = []

    for ticker in unique_tickers:
        try:
            # We fetch a tiny amount of data to check if the ticker is valid
            stock = yf.Ticker(ticker)
            # .history() is a good way to validate; it returns an empty DataFrame for invalid tickers
            if not stock.history(period="5d").empty:
                print(f"✅ SUCCESS: '{ticker}' is a valid yfinance ticker.")
                valid_tickers.append(ticker)
            else:
                print(f"❌ FAILED:  '{ticker}' could not be found by yfinance.")
                invalid_tickers.append(ticker)
        except Exception as e:
            print(f"❌ ERROR:   An error occurred with '{ticker}': {e}")
            invalid_tickers.append(ticker)

    print("\n--- Analysis Complete ---")
    print(f"Valid Tickers: {len(valid_tickers)}")
    print(f"Invalid Tickers: {len(invalid_tickers)}")
    if invalid_tickers:
        print("\nThe following tickers are likely incorrect or need a market suffix (e.g., '.AS'):")
        print(", ".join(invalid_tickers))

if __name__ == "__main__":
    check_tickers()
