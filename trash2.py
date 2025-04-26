import numpy as np
import matplotlib.pyplot as plt

# --- 参数设定 ---
numPixels = 1024              # 输出图像分辨率 1024×1024
pixel_size = 0.4e-6           # 新的像素大小：0.4 μm

# 构造像素坐标索引
x = np.arange(numPixels) - numPixels/2
# 物理坐标（米），直接用新的 pixel_size
x_phys = x * pixel_size

# --- 要生成的光栅周期列表（单位：米） ---
periods = [4e-6, 10e-6, 15e-6, 20e-6, 25e-6, 30e-6]

# --- 逐个生成并保存 ---
for P in periods:
    f = 1.0 / P                                # 物理频率（cycles/m），保持不变
    am_1d = 0.5 + 0.5 * np.sin(2 * np.pi * f * x_phys)
    am_2d = np.tile(am_1d, (numPixels, 1))    # 复制成二维

    # 文件名仍用物理周期标注（微米）
    fname = f'grating_{int(P*1e6):d}um.png'

    # 直接保存，无标题无坐标轴，输出 1024×1024
    plt.imsave(fname, am_2d, cmap='gray', vmin=0, vmax=1)
