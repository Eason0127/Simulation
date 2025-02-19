import numpy as np
import matplotlib.pyplot as plt


# 生成分辨率为 500x500 的显微样本
def generate_microscopic_sample(size=500):
    x, y = np.meshgrid(np.linspace(-5, 5, size), np.linspace(-5, 5, size))

    # 基础结构：高斯斑点（模拟细胞核）
    sample = np.exp(-(x ** 2 + y ** 2))  # 中心高斯斑点

    # 添加随机高斯斑点（模拟细胞器）
    for _ in range(20):
        x0, y0 = np.random.uniform(-4, 4, 2)
        sigma = np.random.uniform(0.1, 0.5)
        sample += 0.5 * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # 添加正弦波纹理（模拟细胞质或周期性结构）
    sample += 0.3 * np.sin(10 * x) * np.sin(10 * y)

    # 添加噪声（模拟显微镜噪声）
    sample += 0.1 * np.random.randn(size, size)

    # 归一化到 [0, 1]
    sample = (sample - sample.min()) / (sample.max() - sample.min())
    return sample


# 生成并显示样本
sample = generate_microscopic_sample()
plt.imshow(sample, cmap='gray', vmin=0, vmax=1)
plt.title("Simulated Microscopic Sample (500x500)")
plt.axis('off')
plt.savefig("microscopic_sample.png", bbox_inches='tight', dpi=300)
plt.show()