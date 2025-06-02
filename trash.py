import numpy as np
import matplotlib.pyplot as plt

# 数据
distances = np.array([0.0020, 0.0019, 0.0018, 0.0017, 0.0016, 0.0015, 0.0014, 0.0013,
                      0.0012, 0.0011, 0.0010, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005,
                      0.0004, 0.0003, 0.0002, 0.0001, 0.00009, 0.00008, 0.00007, 0.00006,
                      0.00005, 0.00004])
resolutions = np.array([11, 11, 11, 11,  9,  9,  8,  8,
                         7,  6,  6,  6,  6,  6,  5,  5,
                         4,  4,  4,  7,  4,  5,  6,  6,
                         6,  6])

# 转为 mm
distances_mm = distances * 1e3

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(distances_mm, resolutions, 'o-', linewidth=2)
plt.xlabel('Sample–Sensor Distance (mm)')
plt.ylabel('Reconstructed Resolution (line group index)')
plt.title('Resolution vs. Distance (pixel size = 1.2 μm)')
plt.grid(True, linestyle='--', alpha=0.7)

# 横坐标从大到小
plt.gca().invert_xaxis()

plt.tight_layout()
plt.show()
