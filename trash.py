import numpy as np
import matplotlib.pyplot as plt

# 给定参数
pitch_size = 3.45e-6
w = pitch_size * 3000
wavelength = 532e-9

# 空间采样频率和采样分辨率（μm）
fs = 1 / (2 * pitch_size)
rho_s = 1 / fs          # m
rho_s_um = rho_s * 1e6  # μm

# z2 从 0 到 300 mm
z2_mm = np.linspace(0, 300, 1000)
z2 = z2_mm * 1e-3

# 计算 NA(z2), fx(z2) 以及成像分辨率 rho_x(z2)=1/fx
NA = (w / 2) / np.sqrt((w / 2)**2 + z2**2)
fx = NA / wavelength
rho_x = 1 / fx          # m
rho_x_um = rho_x * 1e6  # μm

# 交点：fx = fs ⇒ rho_x = rho_s
z_intersect = np.sqrt((w/2)**2 * (1/(fs * wavelength)**2 - 1))
z_int_mm = z_intersect * 1e3

fig, ax = plt.subplots(figsize=(8,5))

# 成像分辨率曲线
ax.plot(z2_mm, rho_x_um, label=r'$\rho_x(z_2)=\frac{1}{f_x(z_2)}$')

# 标出交点及垂线
ax.plot(z_int_mm, rho_s_um, 'o', label='Intersection')
ax.axvline(x=z_int_mm, linestyle=':', linewidth=1)

# 给交点左侧区域打灰色阴影
ax.axvspan(0, z_int_mm, color='gray', alpha=0.3)

# 把 x=0 和 y=0 的轴穿过原点
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# 添加标注：交点的 z2 和 resolution
ax.annotate(
    f'$z_2={z_int_mm:.1f}\\,\\mathrm{{mm}}$\n$\\rho={rho_s_um:.1f}\\,\\mu\\mathrm{{m}}$',
    xy=(z_int_mm, rho_s_um),
    xytext=(z_int_mm + 20, rho_s_um + 5),
    textcoords='data',
    arrowprops=dict(arrowstyle='->', lw=1),
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray')
)

ax.set_xlabel('Propagation distance $z_2$ (mm)')
ax.set_ylabel('Resolution (μm)')
ax.set_title('Resolution vs. $z_2$ (0–300 mm)')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()
