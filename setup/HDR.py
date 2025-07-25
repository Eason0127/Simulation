from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


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


def read_image(path: str) -> np.ndarray:
    img = Image.open(path).convert('L')
    return np.array(img, dtype=float)   # float 方便后续累乘

import numpy as np

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

image = read_image("/Users/wangmusi/Desktop/Research/HDR/sample.jpeg")
plot_image(image)
sat_val = 10000
expo_time = np.arange(10,70,10)
L = 2 ** 6 + 1 # 传感器的位深
levels = np.linspace(0,sat_val,L) # 电信号
imaging_collection = []
for i in expo_time:
    object2 = i * image # 曝光
    mask = object2 > sat_val # 过曝的值调整
    object2[mask] = sat_val
    normalized = np.clip(object2 / sat_val, 0, 1) # 归一化信号
    object_recorded = normalized * (L - 1)# 模拟传感器记录信号
    real_signal = np.round(object_recorded).astype(int) # 四舍五入取整数
    quantized = levels[real_signal] # 讲得到的信号反应为电信号
    print(i)
    plot_image(real_signal)
    imaging_collection.append(real_signal)

HDR = merge_hdr(imaging_collection, expo_time, sat_val)
print("Finished")
plot_image(HDR)
