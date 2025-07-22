import numpy as np
import cv2
import tifffile as tiff
from pathlib import Path
def read_gray(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3:                        # BGR -> Gray
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im

hdr   = tiff.imread(r"C:\Users\GOG\Desktop\Research\HDR2\hdr_float32.tif").astype(np.float32)
img60 = read_gray(r"C:\Users\GOG\Desktop\Research\HDR2\exp_60.03ms.png").astype(np.float32)
dark  = read_gray(r"C:\Users\GOG\Desktop\Research\HDR2\dark_20.02ms.png").astype(np.float32)

tdark = 0.02002
texp  = 0.06003
sat   = 65535
low   = 100

img60c = np.clip(img60 - dark*(texp/tdark), 0, None)
pred60 = hdr * texp

mask = (img60c < sat-500) & (img60c > low)
# 额外保证没有 NaN/Inf
mask &= np.isfinite(img60c) & np.isfinite(pred60)

err = pred60[mask] - img60c[mask]
rel_err = np.median(np.abs(err) / (img60c[mask] + 1e-6))
print("median relative error:", rel_err)

# 线性拟合
p = np.polyfit(pred60[mask], img60c[mask], 1)
print("slope:", p[0], "intercept:", p[1])

