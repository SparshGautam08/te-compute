"""
Minute-level version of the transfer entropy experiment.

Runs the whole pipeline in one go, on 1-minute prices instead of hourly:
  1. Download 1-minute closing prices (BTC, ETH, SOL, DOGE) from Yahoo Finance
  2. Clean them and convert to log returns (same steps as prepare_data.py)
  3. Count ties and add tiny tie-breaking noise (Kraskov et al. 2004)
  4. Same-minute correlations
  5. Transfer entropy for all 12 directed pairs, with a shuffle test

Note: Yahoo only serves about the last week of 1-minute data, so this is a
short window (~10,000 minutes). Treat the results as a preliminary check.

Run from the repository root with:  python scripts/minute_experiment.py
"""

import itertools

import numpy as np
import pandas as pd
import yfinance as yf
from PyIF import te_compute as te

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repository root

# ---------------- Settings ----------------
TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"]
PERIOD = "7d"        # Yahoo limit for 1-minute data is about one week
INTERVAL = "1m"
K = 4                # number of neighbors, used in BOTH te_compute calls
EMBEDDING = 1        # 1 minute of history
N_SHUFFLES = 100
NOISE_SD = 1e-10     # tie-breaking noise (Kraskov et al. 2004, Sec. III.A)

# ---------------- 1. Download ----------------
data = yf.download(TICKERS, period=PERIOD, interval=INTERVAL)
prices = data["Close"]
print("Downloaded:", prices.shape, "from", prices.index.min(), "to", prices.index.max())
print("Blank prices per coin:")
print(prices.isna().sum())
prices.to_csv(ROOT / "data" / "minute" / "prices_minute.csv")

# ---------------- 2. Clean and convert to log returns ----------------
# Drop the last row: the current minute isn't finished yet
prices = prices.iloc[:-1]

# Put every minute on a complete grid so missing minutes become blanks
full_index = pd.date_range(prices.index.min(), prices.index.max(), freq="min")
missing_rows = len(full_index) - len(prices)
print("\nMinutes with no row at all:", missing_rows)
prices = prices.reindex(full_index)

returns = np.log(prices).diff()
n_possible = len(returns) - 1
returns = returns.dropna()
n_dropped = n_possible - len(returns)
print(f"Dropped {n_dropped} of {n_possible} returns "
      f"({100 * n_dropped / n_possible:.2f}%) because of gaps or blanks")
print("Final dataset:", returns.shape)
print(returns.describe())
returns.to_csv(ROOT / "data" / "minute" / "returns_minute.csv")

# ---------------- 3. Ties and tie-breaking noise ----------------
print("\nMinutes with exactly zero return:")
print((returns == 0).sum())
print("Repeated values per coin (ties of any kind):")
print(returns.apply(lambda col: col.duplicated().sum()))

rng = np.random.default_rng(42)
returns_jittered = returns + rng.normal(0, NOISE_SD, size=returns.shape)
returns_jittered.to_csv(ROOT / "data" / "minute" / "returns_minute_jittered.csv")

# ---------------- 4. Same-minute correlations ----------------
print("\nSame-minute correlations:")
print(returns.corr().round(3))

# ---------------- 5. Transfer entropy + shuffle test ----------------
coins = list(returns_jittered.columns)
shuffle_rng = np.random.default_rng(0)
rows = []

for source, target in itertools.permutations(coins, 2):
    y = returns_jittered[target].values
    x = returns_jittered[source].values
    # Verified with direction_test.py: te_compute(target, source) measures source -> target
    real = te.te_compute(y, x, k=K, embedding=EMBEDDING)
    null = [te.te_compute(y, shuffle_rng.permutation(x), k=K, embedding=EMBEDDING)
            for _ in range(N_SHUFFLES)]
    p = (1 + sum(v >= real for v in null)) / (N_SHUFFLES + 1)
    rows.append({"source": source, "target": target, "TE": real,
                 "shuffled_mean": np.mean(null), "shuffled_max": np.max(null),
                 "p": p})
    print(source, "->", target, "done")

results = pd.DataFrame(rows)
print(f"\nTransfer entropy, 1-minute returns, k={K}, {N_SHUFFLES} shuffles:")
print(results.round(4))
print("\nSignificant pairs (p < 0.05):")
print(results[results["p"] < 0.05][["source", "target", "TE", "p"]].round(4))
results.to_csv(ROOT / "results" / "minute" / f"te_shuffle_minute_k{K}.csv", index=False)
