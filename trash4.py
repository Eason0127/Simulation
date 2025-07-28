import numpy as np
from numpy.fft import fft2, fftshift, fftfreq
from PIL import Image
def estimate_image_max_frequency(img: np.ndarray,
                                 pixel_size: float,
                                 energy_thresh: float = 1e-3):
    """
    估计一张灰度图像的最高空间频率。

    参数
    ----
    img : np.ndarray, shape (H, W)
      灰度图像，任意 dtype
    pixel_size : float
      像素间距（单位：米／像素）
    energy_thresh : float
      能量阈值，阈值 = freq_bin_energy / total_energy，
      低于此阈值的频率分量会被视为噪声忽略。

    返回
    ----
    f_nyq : float
      理论奈奎斯特频率，1/(2*pixel_size)
    f_emp : float
      实际在频谱中出现且能量高于阈值的最大频率（周期／米）
    """

    H, W = img.shape

    # 理论最大（奈奎斯特）——水平/垂直方向
    f_nyq = 1.0 / (2.0 * pixel_size)

    # 1) 计算二维 FFT 并移到中心
    F = fftshift(fft2(img))
    mag2 = np.abs(F)**2        # 功率谱

    # 2) 构造频率坐标轴
    fx = fftshift(fftfreq(W, d=pixel_size))  # 水平方向频率数组
    fy = fftshift(fftfreq(H, d=pixel_size))  # 垂直方向频率数组
    FX, FY = np.meshgrid(fx, fy)
    FR = np.sqrt(FX**2 + FY**2)               # 径向频率

    # 3) 阈值过滤：保留能量大于总能量 * energy_thresh 的分量
    total_energy = mag2.sum()
    mask = mag2 >= (total_energy * energy_thresh)

    # 4) 在保留的分量里，找到最大的 FR
    if np.any(mask):
        f_emp = FR[mask].max()
    else:
        f_emp = 0.0

    return f_nyq, f_emp

from PIL import Image

# 读图到 NumPy 数组
img = np.array(Image.open("C:/Users\GOG\Desktop\exp_60.03ms.png").convert("L"), dtype=float)

# 假设像素间距 5.86 μm
pixel_size = 5.86e-6  # 米/像素

f_nyq, f_emp = estimate_image_max_frequency(img, pixel_size, energy_thresh=1e-4)
print(f"理论奈奎斯特频率 = {f_nyq:.1e} 周期/米")
print(f"实际检测到的最高频率 = {f_emp:.1e} 周期/米")
