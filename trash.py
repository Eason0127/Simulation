import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# --- 频域插值 + 低通滤波 ---
def resize_fourier_lowpass(img, new_size):
    old_size = img.shape[0]
    img_ft = fftshift(fft2(ifftshift(img)))  # 计算傅里叶变换

    new_ft = np.zeros((new_size, new_size), dtype=np.complex64)

    center = old_size // 2
    new_center = new_size // 2
    half_size = min(new_center, center)

    new_ft[new_center - half_size : new_center + half_size,
           new_center - half_size : new_center + half_size] = \
          img_ft[center - half_size : center + half_size,
                 center - half_size : center + half_size]

    # 应用高斯低通滤波器
    X, Y = np.meshgrid(np.linspace(-1, 1, new_size), np.linspace(-1, 1, new_size))
    lowpass_filter = np.exp(- (X**2 + Y**2) * 10)  # 调整参数以控制频率范围
    new_ft *= lowpass_filter

    resized_img = fftshift(ifft2(ifftshift(new_ft))).real  # 逆变换
    return resized_img

# --- 定义参数 ---
original_pixels = 1024
original_pixel_size = 0.2e-6
new_pixel_size = 1.6e-6
FOV = original_pixels * original_pixel_size
new_pixels = int(FOV / new_pixel_size)

# --- 读取原始 .npy 文件 ---
input_npy = "original_data.npy"
output_npy = "resized_data.npy"

data = np.load('hologram.npy').astype(np.float32)  # 确保 float32
original_max = np.max(data)
data /= original_max  # 归一化
resized_data = resize_fourier_lowpass(data, new_pixels)  # 采用低通傅里叶插值
resized_data *= original_max  # 反归一化

# --- 确保数据格式正确并保存 ---
resized_data = np.squeeze(resized_data).astype(np.float32)  # 确保 2D
np.save(output_npy, resized_data)
