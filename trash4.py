import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def read_image(path: str) -> np.ndarray:
    img = Image.open(path).convert('L')
    return np.array(img, dtype=float)

def plot_difference(img1: np.ndarray, img2: np.ndarray, cmap='seismic'):
    """
    绘制两幅图像的差异图 (img1 - img2)

    参数:
        img1, img2 : 两幅灰度图 (numpy array, shape 相同)
        cmap       : 差异 colormap，默认 'seismic' (红蓝对比)
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes differ: {img1.shape} vs {img2.shape}")

    diff = img1 - img2

    plt.imshow(diff, cmap=cmap)
    plt.colorbar(label="Difference")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return diff

o1 = read_image("/Users/wangmusi/Desktop/normalrec.png")
o2 = read_image("/Users/wangmusi/Desktop/HDRrec.png")
plot_difference(o1, o2)