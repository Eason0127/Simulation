import numpy as np
import matplotlib.pyplot as plt

# 参数
w = 10350e-6       # 采样宽度 409.6 μm
wavelength = 532e-9  # 波长 532 nm

# z2 取值范围 1 mm 至 10 mm
z2 = np.linspace(1e-3, 10e-2, 1000)

# 计算 NA 和 f_cut
NA = (w / 2) / np.sqrt((w / 2)**2 + z2**2)
f_cut = NA / wavelength

# 计算分辨率 1/(2*f_cut)，并转换为微米
resolution = 1 / (2 * f_cut)
resolution_um = resolution * 1e6

# 标记每隔 1 mm 的点
z2_marks = np.arange(1e-3, 1.01e-2, 1e-3)
NA_marks = (w / 2) / np.sqrt((w / 2)**2 + z2_marks**2)
f_cut_marks = NA_marks / wavelength
resolution_marks = 1 / (2 * f_cut_marks)
resolution_marks_um = resolution_marks * 1e6

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(z2 * 1e3, resolution_um, label=r'$1/(2 f_{\mathrm{cut}})$')
plt.scatter(z2_marks * 1e3, resolution_marks_um, marker='o', label='标记点')

for x, y in zip(z2_marks * 1e3, resolution_marks_um):
    plt.annotate(f'{int(x)} mm', xy=(x, y), xytext=(0, 5),
                 textcoords='offset points', ha='center', fontsize=8)

plt.xlabel('Sample-sensor distance $z_2$ (mm)')
plt.ylabel('Resolution $1/(2f_{\\mathrm{cut}})$ (μm)')
plt.title('Resolution $1/(2 f_{\\mathrm{cut}})$ vs. $z_2$ (w=409.6 μm, λ=532 nm)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
