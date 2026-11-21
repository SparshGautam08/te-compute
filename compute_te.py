import itertools
import pandas as pd
from PyIF import te_compute as te

returns = pd.read_csv("returns_hourly_jittered.csv", index_col=0)
coins = list(returns.columns)

rows = []
# Every ordered pair: (BTC, ETH), (ETH, BTC), ... 12 in total
for source, target in itertools.permutations(coins, 2):
    # Verified earlier: te_compute(target, source) measures source -> target
    value = te.te_compute(returns[target].values, returns[source].values,
                          k=1, embedding=1)
    rows.append({"source": source, "target": target, "TE": value})

results = pd.DataFrame(rows)
print(results.pivot(index="source", columns="target", values="TE").round(4))
results.to_csv("te_results.csv", index=False)