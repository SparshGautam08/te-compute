import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repository root

prices = pd.read_csv(ROOT / "data" / "hourly" / "prices_hourly.csv", index_col=0, parse_dates=True)

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

returns.to_csv(ROOT / "data" / "hourly" / "returns_hourly.csv")

# How many values repeat an earlier value (ties of any kind)?
print("Repeated values per coin:")
print(returns.apply(lambda col: col.duplicated().sum()))

# add noise to break ties
rng = np.random.default_rng(42)   # fixed seed so results are repeatable
noise = rng.normal(0, 1e-10, size=returns.shape)
returns_jittered = returns + noise

returns_jittered.to_csv(ROOT / "data" / "hourly" / "returns_hourly_jittered.csv")