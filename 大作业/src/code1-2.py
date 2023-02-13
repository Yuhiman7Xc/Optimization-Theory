import numpy as np
import random
import matplotlib.pyplot as plt

# ASize = (10, 300)
# XSize = 300
# # A 为 10ⅹ300 维的测量矩阵，A 中的元素服从均值为 0 方差为 1 的高斯分布。
# A = np.random.normal(0, 1, ASize)
# # x 为 300 维的未知稀疏向量且稀疏度为 5
# X = np.zeros(XSize)
# # e 为 10 维的测量噪声，e 中的元素服从均值为 0 方差为 0.2 的高斯分布。
# e = np.random.normal(0, 0.2, 10)

ASize = (50, 100)
XSize = 100
A = np.random.normal(0, 1, ASize)
X = np.zeros(XSize)
e = np.random.normal(0, 0.2, 50)

XIndex = random.sample(list(range(XSize)), 5)  # 5 稀疏度
for xi in XIndex:
    X[xi] = np.random.randn()

b = np.dot(A, X) + e

np.save("A.npy", A)
np.save("X.npy", X)
np.save("b.npy", b)

# 交替反向乘子法

# A = np.load('A.npy')
# b = np.load('b.npy')
# X = np.load('X.npy')

# ASize = (10, 300)
# BSize = 10
# XSize = 300

P = 10
c = 0.005
Xk = np.zeros(XSize)
Zk = np.zeros(XSize)
Vk = np.zeros(XSize)

X_opt_dst_steps = []
X_dst_steps = []

while True:
    Xk_new = np.dot(
        np.linalg.inv(np.dot(A.T, A) + c * np.eye(XSize, XSize)),
        c*Zk + Vk + np.dot(A.T, b)
    )
    # 软门限算子
    Zk_new = np.zeros(XSize)
    for i in range(XSize):
        if Xk_new[i] - Vk[i] / c < - P / c:
            Zk_new[i] = Xk_new[i] - Vk[i] / c + P / c
        elif Xk_new[i] - Vk[i] / c > P / c:
            Zk_new[i] = Xk_new[i] - Vk[i] / c - P / c

    Vk_new = Vk + c * (Zk_new - Xk_new)

    # print(np.linalg.norm(Xk_new - Xk, ord=2))

    X_dst_steps.append(np.linalg.norm(Xk_new - X, ord=2))
    X_opt_dst_steps.append(Xk_new)
    if np.linalg.norm(Xk_new - Xk, ord=2) < 1e-5:
        break
    else:
        Xk = Xk_new.copy()
        Zk = Zk_new.copy()
        Vk = Vk_new.copy()

print(Xk)
print(X)

X_opt = X_opt_dst_steps[-1]

for i, data in enumerate(X_opt_dst_steps):
    X_opt_dst_steps[i] = np.linalg.norm(data - X_opt, ord=2)
plt.title("Distance")
plt.plot(X_opt_dst_steps, label='X-optimal-distance')
plt.plot(X_dst_steps, label='X-real-distance')
plt.grid()
plt.legend()
plt.show()

