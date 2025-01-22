import numpy as np
from PIL import Image, ImageDraw

def generate_sine_image(width, height, frequency, amplitude, line_thickness, output_file):
    # 创建空白画布
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # 中心线位置
    center_y = height // 2

    # 生成正弦曲线并绘制
    for x in range(width):
        y = int(center_y + amplitude * np.sin(2 * np.pi * frequency * x / width))
        for t in range(-line_thickness // 2, line_thickness // 2 + 1):
            if 0 <= y + t < height:
                canvas[y + t, x] = [255, 255, 255]  # 白色线条

    # 保存图像
    image = Image.fromarray(canvas)
    image.save(output_file)

# 示例：生成1024x1024的正弦函数图像
width = 1024
height = 1024
frequency = 5  # 频率
amplitude = 100  # 振幅
line_thickness = 20  # 线条粗细
output_file = "sine_wave2.png"

generate_sine_image(width, height, frequency, amplitude, line_thickness, output_file)
