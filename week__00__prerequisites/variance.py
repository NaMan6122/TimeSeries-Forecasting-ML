import numpy as np

z = np.random.randn(1000)
var_z = np.var(z);

# creating two independent series.
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)

cov_xy = np.cov(x, y)[0,1]
print(f"Covariance (independent): {cov_xy:.4f}")   # near 0

# sim lag 1 autocovariance, basically shifting the data by '1' step (K), refer notes.
x_lag1 = x[1:]
x_original = x[:-1]

cov_auto = np.cov(x_original, x_lag1)[0,1]
print(f"Autocovariance (lag 1): {cov_auto:.4f}")    # not zero by chance

y = 2 * x + np.random.randn(100) * 0.1 # Sim for High CoVariance.
print(np.cov(x, y)[0,1])
