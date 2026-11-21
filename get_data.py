import yfinance as yf

# The four coins, as Yahoo Finance names them
tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"]

# Download hourly prices for the last ~2 years
data = yf.download(tickers, period="720d", interval="1h")

# Keep only the closing price of each hour
prices = data["Close"]

# Check what we got
print(prices.head())         # first 5 rows
print(prices.tail())         # last 5 rows
print(prices.shape)          # (number of rows, number of columns)
print(prices.isna().sum())   # missing values per coin

# Save to a file so we don't have to re-download
prices.to_csv("prices_hourly.csv")