import yfinance as yf
data = yf.download("AAPL", period="6mo")
print(data.head())
