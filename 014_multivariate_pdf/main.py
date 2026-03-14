import numpy as np
from scipy.stats import multivariate_normal as mvnorm



x = np.random.rand(5)
y = np.random.rand(5)

cov = np.eye(5)

norm = mvnorm(x, cov)

print(norm.pdf(y))

