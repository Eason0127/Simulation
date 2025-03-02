import matplotlib.pyplot as plt

# 像素尺寸（横坐标）
pixel_sizes = [0.2, 1.6]

# Sample 2 Without Noise
rms_wo_noise  = [0.0806, 0.0980]
ssim_wo_noise = [0.8623, 0.7123]

# Sample 2 With Noise
rms_w_noise   = [0.0808, 0.0990]
ssim_w_noise  = [0.8614, 0.7115]

# 创建画布与第一个坐标轴（用于绘制 RMS）
fig, ax1 = plt.subplots(figsize=(6, 4))

# 在同一张图上创建第二个坐标轴（用于绘制 SSIM）
ax2 = ax1.twinx()

# 绘制 RMS (左 y 轴)
line1 = ax1.plot(pixel_sizes, rms_wo_noise, color='blue', marker='^',
                 label='RMS (Without Noise)')
line2 = ax1.plot(pixel_sizes, rms_w_noise,  color='green', marker='^',
                 label='RMS (With Noise)')

# 绘制 SSIM (右 y 轴)
line3 = ax2.plot(pixel_sizes, ssim_wo_noise, color='red', marker='s',
                 label='SSIM (Without Noise)')
line4 = ax2.plot(pixel_sizes, ssim_w_noise,  color='purple', marker='s',
                 label='SSIM (With Noise)')

# 设置坐标轴标题
ax1.set_xlabel('Pixel Size (μm)')
ax1.set_ylabel('RMS')
ax2.set_ylabel('SSIM')

# 设置图表标题
plt.title('Sample 2: RMS & SSIM (With/Without Noise)')

# 组合图例（因为有双轴，需要把各自的线条收集起来）
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='best')

# 开启网格（仅对第一个轴有效，若想对第二个轴也加网格可单独设置）
ax1.grid(True)

plt.tight_layout()
plt.show()
