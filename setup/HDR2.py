from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
def read_image(path: str) -> np.ndarray:
    img = Image.open(path).convert('L')
    return np.array(img, dtype=float)   # float 方便后续累乘

def plot_image(img):
    """
    将一个 PIL.Image.Image 或者 NumPy 数组绘制出来。

    参数:
        img: 要绘制的图像，可以是
             - PIL.Image.Image 对象
             - NumPy 数组 (H, W) 或 (H, W, C)
    """
    # 如果是 PIL 对象，先转成数组
    if isinstance(img, Image.Image):
        arr = np.array(img)
    else:
        arr = img

    # 如果是灰度二维数组，就用灰度 colormap；否则按默认显示彩色
    if arr.ndim == 2:
        plt.imshow(arr, cmap='gray')
    else:
        plt.imshow(arr)

    plt.axis('off')  # 不显示坐标轴
    plt.tight_layout()
    plt.show()
def merge_hdr(quantized_list, expo_times, sat_val):
    # 准备累加器
    H, W = quantized_list[0].shape
    numerator   = np.zeros((H, W), dtype=np.float64)
    denominator = np.zeros((H, W), dtype=np.float64)

    # 定义三角形权重函数
    def weight(q):
        mid = sat_val / 2.0
        # 两侧线性上升／下降
        w = np.where(q <= mid, q, sat_val - q)
        # 防止边缘为零权重，可加个小常数 eps
        return np.clip(w, a_min=1e-4, a_max=None)

    # 遍历所有曝光
    for Q, t in zip(quantized_list, expo_times):
        w = weight(Q)                     # 权重
        E = Q / float(t)                  # 单张的辐射度估计
        numerator   += w * E              # 加权累加
        denominator += w

    hdr = numerator / denominator        # 归一化得到最终辐射度
    return hdr
def save_image(hdr: np.ndarray, out_path: str):
    """
    1) 线性归一化 hdr 到 [0,255]
    2) pad 成正方形，默认在上下左右居中
    3) 存成 PNG（也可改 .jpg/.tif）
    """
    # 1) 归一化
    lo, hi = hdr.min(), hdr.max()
    norm = (hdr - lo) / (hi - lo)
    uint8 = np.round(norm * 255).astype(np.uint8)

    # 2) pad 到方形
    h, w = uint8.shape
    S = max(h, w)
    pad_h = (S - h) // 2
    pad_w = (S - w) // 2
    padded = np.pad(
        uint8,
        ((pad_h, S - h - pad_h), (pad_w, S - w - pad_w)),
        mode='constant',
        constant_values=0
    )

    # 3) 存盘
    img = Image.fromarray(padded, mode='L')
    img.save(out_path)

expo_time = [20.02,29.98,40.02,49.99,60.03,69.99,80.03,89.99,100.04,110,119.96] #ms
sat_value = 255
image_collection = []
for i in expo_time:
    pic = read_image(fr"C:\Users\GOG\Desktop\Research\HDR2\exp_{i}ms.png")
    image_collection.append(pic)
    print(i)
HDR = merge_hdr(image_collection, expo_time, sat_value)
plot_image(HDR)
save_image(HDR,"C:/Users\GOG\Desktop\hdr_sample.png")
# np.save(r"C:\Users\GOG\Desktop\hdr_sample2.npy", HDR)

