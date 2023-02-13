import numpy as np
import random
ASize = (50, 100)
XSize = 100
A = np.random.normal(0, 1, ASize)
X = np.zeros(XSize)
e = np.random.normal(0, 0.1, 50)
XIndex = random.sample(list(range(XSize)), 5)  # 5 稀疏度
for xi in XIndex:
    X[xi] = np.random.randn()

b = np.dot(A, X) + e

np.save("A.npy", A)
np.save("X.npy", X)
np.save("b.npy", b)
