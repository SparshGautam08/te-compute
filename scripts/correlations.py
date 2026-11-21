import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repository root

returns = pd.read_csv(ROOT / "data" / "hourly" / "returns_hourly.csv", index_col=0)

# Same-hour correlation between every pair of coins
print(returns.corr().round(3))