import itertools
import numpy as np
import pandas as pd
from PyIF import te_compute as te

returns = pd.read_csv("returns_hourly_jittered.csv", index_col=0)
coins = list(returns.columns)
rng = np.random.default_rng(0)
n_shuffles = 100
K = 4

rows = []
for source, target in itertools.permutations(coins, 2):
    y = returns[target].values
    x = returns[source].values
    real = te.te_compute(y, x, k=K, embedding=1)
    # Scramble the source's time order, recompute, repeat
    null = [te.te_compute(y, rng.permutation(x), k=K, embedding=1)
            for _ in range(n_shuffles)]
    # p-value: share of shuffles that matched or beat the real value
    p = (1 + sum(v >= real for v in null)) / (n_shuffles + 1)
    rows.append({"source": source, "target": target, "TE": real,
                 "shuffled_mean": np.mean(null), "shuffled_max": np.max(null),
                 "p": p})
    print(source, "->", target, "done")

results = pd.DataFrame(rows)
print(results.round(4))
results.to_csv(f"te_shuffle{K}.csv", index=False)
