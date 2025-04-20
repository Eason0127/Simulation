import numpy as np
import matplotlib.pyplot as plt
import os

# 图像参数
img_size = 1024  # 整幅图像大小为 1024x1024 像素
stripe_width = 60  # 黑色条纹固定宽度 60 像素
gaps = np.arange(10, 35)  # gap 值从 10 到 25（包含25），步长为 1

# 输出文件夹：存放生成的图像
output_folder = "output_gratings"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 对于每一个 gap 值，生成一幅单独的图像
for gap in gaps:
    # 创建一个白色背景图像
    image = np.full((img_size, img_size), 255, dtype=np.uint8)

    # 从最左侧开始生成光栅图案，整个图像纵向都填充相同的模式
    pos = 0
    while pos < img_size:
        pos_end = pos + stripe_width
        pos_end = min(pos_end, img_size)  # 防止超出图像宽度
        # 对整幅图像的所有行，在对应区域填充黑色（像素值 0）
        image[:, pos:pos_end] = 0
        pos += stripe_width + gap

    # 生成图像并保存
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.axis('off')

    # 构造保存文件的路径，文件名中包含 gap 值
    filename = os.path.join(output_folder, f"grating_gap_{gap}.png")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭当前图，释放内存
    print(f"Saved image: {filename}")
