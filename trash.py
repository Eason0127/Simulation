import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from PIL import Image
import os

def load_image_grayscale(path):
    """
    从文件读取图像并转换到 float 灰度数组。
    """
    img = Image.open(path).convert('L')
    return np.array(img, dtype=float)

def calculate_psd(image):
    """
    Calculates the 1D radially averaged amplitude spectrum (PSD-like).
    输入必须是一个 2D numpy 数组（灰度图）。

    Returns:
        r: 半径 bin 中心 (pixels)
        psd_1d: 每个半径 bin 的平均幅值
    """
    if image.ndim != 2:
        raise ValueError("输入必须是一个 2D 数组（灰度图）。")

    M, N = image.shape

    # 2D Hanning 窗，减少谱泄露
    hann_window = np.outer(np.hanning(M), np.hanning(N))
    windowed = image * hann_window

    # FFT 并把零频搬到中心
    F = fftshift(fft2(windowed))
    amplitude = np.abs(F)

    # 计算每个像素到中心的半径
    cy, cx = M//2, N//2
    y, x = np.indices((M, N))
    radii = np.sqrt((x - cx)**2 + (y - cy)**2)

    # 按半径 bin 做 histogram 加权求和
    maxr = int(radii.max())
    bins = np.arange(0, maxr + 1, 1)
    sums, edges = np.histogram(radii.ravel(), bins=bins, weights=amplitude.ravel())
    counts, _ = np.histogram(radii.ravel(), bins=bins)

    # 平均幅值（PSD）
    with np.errstate(divide='ignore', invalid='ignore'):
        psd_1d = sums / counts
    psd_1d[counts == 0] = 0

    # bin 中心
    r = 0.5 * (edges[:-1] + edges[1:])
    return r, psd_1d

def plot_psd(r, psd, label=None):
    plt.loglog(r[1:], psd[1:], label=label)  # 跳过 DC 分量和 r=0
    plt.xlabel("Spatial Frequency (pixels)")
    plt.ylabel("Amplitude")
    if label:
        plt.legend()
    plt.grid(True, which="both", ls="--")

if __name__ == "__main__":
    # —— 示例调用 —— #
    img1 = load_image_grayscale(r"C:\Users\GOG\Desktop\exp_60.03ms.png")
    img2 = load_image_grayscale(r"C:\Users\GOG\Desktop\hdr_sample.png")

    r1, psd1 = calculate_psd(img1)
    r2, psd2 = calculate_psd(img2)

    plt.figure(figsize=(8,6))
    plot_psd(r1, psd1, label="exp_60.03ms")
    plot_psd(r2, psd2, label="hdr_sample")
    plt.title("Radially Averaged Amplitude Spectra Comparison")
    plt.show()
