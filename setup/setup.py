import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fft, fftfreq
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from scipy.ndimage import gaussian_filter

# --- Read image and normalization ---
def load_and_normalize_image(filepath):
    image = Image.open(filepath).convert('L')
    grayscale_data = np.array(image, dtype=np.float32)
    return (grayscale_data - grayscale_data.min()) / (grayscale_data.max() - grayscale_data.min())

# --- Plot image ---

def plot_image2(amplitude, title):
    # 转成 float，防止整数截断
    amp = amplitude.astype(np.float32)
    # 归一化到 [0,1]
    amp = (amp - amp.min()) / (amp.max() - amp.min())

    plt.figure(figsize=(6, 6))
    # 强制显示范围为 [0,1]
    im = plt.imshow(amp, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(im, label="Normalized amplitude")
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_image(amplitude,title, save_dir, pixel, picture):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(amplitude, cmap='gray')
    fig.colorbar(im, ax=ax, label="Amplitude")
    ax.set_title(title)
    ax.axis('off')
    # —— 保存 ——
    if save_dir is not None and pixel is not None and picture is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{pixel:.2f}-{picture:.2f}.png"
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ 图像已保存到：{save_path}")
    else:
        # 如果不满足保存条件，仅关闭 figure
        plt.close(fig)

def update_support_absorption_simple(field_obj, sigma=3, alpha=0.2):
    """
    简化版 Shrink-Wrap：
      - 以振幅偏离 1 的程度为特征
      - 用高斯滤波平滑，再取相对阈值
      - 不做任何形态学清理
    """
    # 1) 振幅偏离度
    amp = np.abs(field_obj)
    dev = np.abs(amp - 1.0)            # 背景振幅 = 1

    # 2) 高斯平滑
    dev_blur = gaussian_filter(dev, sigma=sigma)

    # 3) 相对阈值
    T = alpha * dev_blur.max()
    S = dev_blur > T                   # True = 样本区域
    return S


def Transfer_function(W, H, distance, wavelength, pixelSize, numPixels):
    FX = W / (pixelSize * numPixels)
    FY = H / (pixelSize * numPixels)
    k = 2 * np.pi / wavelength
    square_root = np.sqrt(1 - (wavelength ** 2 * FX ** 2) - (wavelength ** 2 * FY ** 2))
    valid_mask = (wavelength ** 2 * FX ** 2 + wavelength ** 2 * FY ** 2) <= 1
    square_root[~valid_mask] = 0
    temp = np.exp(1j * k * distance * square_root)
    return temp

# --- Angular spectrum method ---
def angular_spectrum_method(field, pixelSize, distance, W, H, numPixels):
    GT = fftshift(fft2(ifftshift(field)))
    transfer = Transfer_function(W, H, distance, 525e-9, pixelSize, numPixels)
    gt_prime = fftshift(ifft2(ifftshift(GT * transfer)))
    return gt_prime
# --- IPR ---
def IPR(Measured_amplitude, distance, k_max, pixelSize, W, H, numPixels):
    update_phase = []
    last_field = None
    for k in range(k_max):
        # a) Sensor plane
        if k == 0:
            phase0 = np.zeros(Measured_amplitude.shape)
            field1 = Measured_amplitude * np.exp(1j * phase0)
        else:
            field1 = Measured_amplitude * np.exp(1j * update_phase[k - 1])
        # b) Backpropagation and apply constraint
        field2 = angular_spectrum_method(field1, pixelSize, -distance, W, H, numPixels)
        phase_field2 = np.angle(field2)  # phase
        amp_field2 = np.abs(field2)  # amplitude
        abso = -np.log(amp_field2 + 1e-8) #1e-8 to prevent 0 value
        # Apply constraints
        abso[abso < 0] = 0
        phase_field2[abso < 0] = 0
        amp_field2 = np.exp(-abso)
        field22 = amp_field2 * np.exp(1j * phase_field2)
        # c) Forward propagation
        field3 = angular_spectrum_method(field22, pixelSize, distance, W, H, numPixels)
        phase_field3 = np.angle(field3)
        update_phase.append(phase_field3)

        # d) Backpropagate to get the image
        field4 = angular_spectrum_method(field3, pixelSize, -distance, W, H, numPixels)
        last_field = field4
    return last_field

object_intensity = load_and_normalize_image("C:\Phythonthings\github\Simulation\setup_pic\hologram.png") # Read the image
background_intensity = load_and_normalize_image("C:\Phythonthings\github\Simulation\setup_pic/background.png") # Read the background
# 加个小常数防止除零
eps = 1e-6
ratio = object_intensity / (background_intensity + eps)
# 归一化比值到合理范围
ratio = np.clip(ratio, 0, np.percentile(ratio, 99))
# 最终用来迭代的振幅
measured_amplitude = np.sqrt(ratio)

Pitch_size = 5.86e-6
z2 = 9.5e-2
num_pixel = 1216

x = np.arange(num_pixel) - num_pixel / 2 - 1
y = np.arange(num_pixel) - num_pixel / 2 - 1
W, H = np.meshgrid(x, y)

rec_field = IPR(measured_amplitude,z2,50,Pitch_size,W,H,num_pixel)
am_rec = np.abs(rec_field)
plot_image2(am_rec,"rec")


