import numpy as np
import pandas as pd

prices = pd.read_csv("prices_hourly.csv", index_col=0, parse_dates=True)

# Drop the last row (current hour isn't finished)
prices = prices.iloc[:-1]

# Put every hour on a complete grid,  missing hours become blanks
full_index = pd.date_range(prices.index.min(), prices.index.max(), freq="h")
print("Hours with no row at all:", len(full_index) - len(prices))
prices = prices.reindex(full_index)

# Log returns: log(price this hour) - log(price last hour)
returns = np.log(prices).diff()

# Drop any hour where a coin's return couldn't be computed
returns = returns.dropna()

print(returns.shape)
print(returns.describe())
print("Hours with exactly zero return:")
print((returns == 0).sum())

returns.to_csv("returns_hourly.csv")
