import numpy as np
from PyIF import te_compute as te

rng = np.random.default_rng(0)   # fixed seed, so results are repeatable
n = 5000

x = rng.normal(size=n)           # x is pure random noise
y = np.zeros(n)
# y at each step = 0.8 * x one step earlier + its own noise
y[1:] = 0.8 * x[:-1] + 0.6 * rng.normal(size=n - 1)

print("te_compute(x, y):", te.te_compute(x, y, k=1, embedding=1))
print("te_compute(y, x):", te.te_compute(y, x, k=1, embedding=1))