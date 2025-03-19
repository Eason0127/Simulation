import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. 生成 1024x1024 的图像，并在 4x4 网格内生成小的不规则图形
height, width = 1024, 1024
image = np.zeros((height, width), dtype=np.uint8)

grid_rows, grid_cols = 4, 4
cell_height, cell_width = height // grid_rows, width // grid_cols

for i in range(grid_rows):
    for j in range(grid_cols):
        # 当前网格的左上角坐标
        x0, y0 = j * cell_width, i * cell_height

        # 随机生成 3~6 个顶点的不规则多边形
        num_vertices = np.random.randint(3, 7)

        # 设置边距，避免图形过大
        margin_x = int(cell_width * 0.2)
        margin_y = int(cell_height * 0.2)

        pts = []
        for _ in range(num_vertices):
            x = np.random.randint(x0 + margin_x, x0 + cell_width - margin_x)
            y = np.random.randint(y0 + margin_y, y0 + cell_height - margin_y)
            pts.append([x, y])
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))

        # 填充多边形区域，图形内灰度为 1
        cv2.fillPoly(image, [pts], 1)

# 2. 提取图形部分（非零区域）的最小外接矩形
coords = np.column_stack(np.where(image > 0))
if coords.size == 0:
    # 若图像中没有非零像素，则保持原图
    cropped = image
else:
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1  # 包含最后一个像素
    cropped = image[y_min:y_max, x_min:x_max]

# 3. 保存裁剪后的图像到文件（注意保存时将灰度 1 映射为 255）
cv2.imwrite('pic/cropped_image.png', cropped * 255)

# 显示裁剪后的图像
plt.figure(figsize=(6, 6))
plt.imshow(cropped, cmap='gray', vmin=0, vmax=1)
plt.title("Cropped Image")
plt.axis('off')
plt.show()
