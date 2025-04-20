import numpy as np
import matplotlib.pyplot as plt

# 参数设置
f = 1
x = np.linspace(0, 2, 1000)  # x 在区间 [0,2] 内取 1000 个点
alpha = 0.5 + 0.5 * np.sin(2 * np.pi * f * x)

# 绘制图像
plt.figure(figsize=(8, 4))
plt.plot(x, alpha, label=r'$\alpha = 0.5 + 0.5\sin(2\pi f x)$')
plt.xlabel('x')
plt.ylabel(r'Absorption')
plt.legend()
plt.grid(True)
plt.show()
