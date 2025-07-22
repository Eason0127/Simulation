import numpy as np
import cv2
import tifffile as tiff
from pathlib import Path

# ========= 需要你修改的部分 =========
files = [
    (r"C:\Users\GOG\Desktop\Research\HDR2\b_20.02.png", 0.02002),
    (r"C:\Users\GOG\Desktop\Research\HDR2\b_29.98.png", 0.02998),
    (r"C:\Users\GOG\Desktop\Research\HDR2\b_40.02.png", 0.04002),
    (r"C:\Users\GOG\Desktop\Research\HDR2\b_49.99.png", 0.04999),
    (r"C:\Users\GOG\Desktop\Research\HDR2\b_60.03.png", 0.06003),
]
dark_path = r"C:\Users\GOG\Desktop\Research\HDR2\dark_20.02ms.png"  # 可空
tdark = 0.02002  # 暗场曝光时间（秒），若不同请改
low_clip = 100

output_path = r"C:\Users\GOG\Desktop\Research\HDR2\background.tif"

# ========= 工具函数 =========
def read_any(fp):
    """以保持位深的方式读图；返回单通道 ndarray"""
    ext = Path(fp).suffix.lower()
    if ext in (".tif", ".tiff"):
        im = tiff.imread(fp)
    else:
        im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)  # 保持原始深度
        if im is None:
            raise FileNotFoundError(f"Cannot read {fp}")
        # 如果是彩色，取单通道（一般科研都是灰度；如需别的处理请改）
        if im.ndim == 3:
            # OpenCV 默认 BGR；取任意一通道或转灰度
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im

def compute_weights(img_stack, sat, low):
    # img_stack: [N, H, W]
    x = (img_stack - low) / (sat - low)
    x = np.clip(x, 0, 1)
    w = x * (1 - x)
    w[img_stack >= (sat - 1)] = 0
    return w

# ========= 读取数据 =========
imgs = []
times = []
for fp, t in files:
    img = read_any(fp).astype(np.float32)
    imgs.append(img)
    times.append(t)
imgs = np.stack(imgs, axis=0)  # [N,H,W]
times = np.array(times, dtype=np.float32)

# 自动设定饱和值
if imgs.dtype == np.float32:
    # 如果来源本来就是 float，则你应自己指定饱和阈值
    saturation_level = 65535.0
else:
    saturation_level = float(np.iinfo(imgs.dtype).max)

# ========= 暗场校正 =========
if Path(dark_path).exists():
    dark = read_any(dark_path).astype(np.float32)
    dark_stack = np.stack([dark * (ti / tdark) for ti in times], axis=0)
    imgs = imgs - dark_stack
    imgs = np.clip(imgs, 0, None)

# ========= 权重 & 合成 =========
weights = compute_weights(imgs, saturation_level, low_clip)
inv_times = (1.0 / times)[:, None, None]
numerator = np.sum(weights * imgs * inv_times, axis=0)
denominator = np.sum(weights, axis=0)
valid = denominator > 0
hdr = np.zeros_like(numerator, dtype=np.float32)
hdr[valid] = numerator[valid] / denominator[valid]

# ========= 保存 =========
tiff.imwrite(output_path, hdr.astype(np.float32))
print(f"HDR saved to {output_path}, dtype={hdr.dtype}, range=({hdr.min()}, {hdr.max()})")
