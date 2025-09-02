# -*- coding: utf-8 -*-
"""
分辨率 Δx 随样品-传感器距离 z2 的变化曲线
依据图中给出的判据与参数
"""

import numpy as np
import matplotlib.pyplot as plt

# ==== 可选：如果中文字体报错，取消下面两行注释并安装对应字体 ====
# plt.rcParams['font.sans-serif'] = ['SimHei']    # 或者 'Noto Sans CJK SC'
# plt.rcParams['axes.unicode_minus'] = False

# ----- 已知参数（单位见注释） -----
delta_pixel = 5.86       # 像元尺寸 Δ = 1 µm
w = 3516                # 传感器宽度 w = 240 µm
lam = 0.525              # 波长 λ = 0.532 µm (532 nm)
n = 1.0                  # 折射率 n = 1
delta_lambda = 0.03      # LED 带宽 Δλ = 0.03 µm (30 nm)
z1_mm = 220.0            # LED-样品距离 z1 = 20 cm = 200 mm
z1 = z1_mm * 1000.0      # 转成 µm
D = 50.0                 # 有效光源尺寸 D = 50 µm
z2_mm = np.linspace(60.0, 120.0, 1200)
z2 = z2_mm * 1000.0  # µm
# ----- 派生量 -----
# Gaussian 光谱的纵向相干长度：Lc = (2 ln 2 / π) * λ^2 / Δλ
L_coh = (2 * np.log(2) / np.pi) * (lam**2 / delta_lambda)
         # 纵向相干长度 ≈ λ^2/Δλ (µm)
r_c = 0.5 * lam * (z1+z2) / D               # 空间相干半径 (µm)

# ----- 距离范围：z2 = 1 ~ 3 mm -----


# ----- 各限制角 -----
theta1 = np.arccos(z2 / (z2 + L_coh))       # 时间相干限制
theta2 = np.arctan(r_c / z2)                # 空间相干/相干半径限制
theta_geo = np.arctan((w/2.0) / z2)         # 传感器几何视场限制

theta_max = np.minimum.reduce([theta1, theta2, theta_geo])

# ----- 分辨率 -----
delta_x = lam / (2.0 * n * np.sin(theta_max))          # 衍射/相干限制的分辨率 (µm)
final_resolution = np.maximum(delta_x, delta_pixel)    # 考虑像元尺寸

# ====== 你的测量数据（红色叉叉） ======
data_d_mm = np.array([
    68, 70, 71, 72, 75, 76, 77, 80, 81, 82, 84, 85, 87, 92, 93, 96, 100,110, 119
])
data_res_um = np.array([
    24.8, 25, 25.3, 25.5, 25.8, 26, 26.3, 26.6, 26.8, 27, 27.2, 27.5,27.8, 28.5, 28.8, 29.5, 31.05, 33,35
])

# ----- 绘图 -----
plt.figure(figsize=(6, 4))
plt.plot(z2_mm, final_resolution, label='Theoretical resolution curve')
# 加入红色叉叉的数据点
plt.plot(
    data_d_mm, data_res_um,
    linestyle='None', marker='x', markersize=8, markeredgewidth=1.8,
    color='red', label='Test data'
)

plt.xlabel('Sample-sensor distance z₂ (mm)')
plt.ylabel('Resolution (µm)')
# plt.title('2µm')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----- 可选：打印几个代表值 -----
for mm in [1, 2, 3]:
    i = np.argmin(np.abs(z2_mm - mm))
    print(f"z2 = {mm:.0f} mm -> Δx(最终) ≈ {final_resolution[i]:.2f} µm")
