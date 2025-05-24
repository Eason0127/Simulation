from PIL import Image, ImageDraw

# 创建一个 1024x1024 的白色画布
img = Image.new('RGB', (1024, 1024), 'white')
draw = ImageDraw.Draw(img)

# 圆的参数
radius = 5       # 半径 = 5 像素（直径 = 10 像素）
spacing = 14     # 两个圆边缘间距 = 50 像素

# 计算水平居中时两个圆心的坐标
total_width = 2 * (2 * radius) + spacing
left_edge = (1024 - total_width) / 2
center_x1 = left_edge + radius
center_x2 = center_x1 + 2 * radius + spacing
center_y = 1024 / 2

# 画第一个圆
draw.ellipse(
    (center_x1 - radius, center_y - radius,
     center_x1 + radius, center_y + radius),
    fill='black'
)

# 画第二个圆
draw.ellipse(
    (center_x2 - radius, center_y - radius,
     center_x2 + radius, center_y + radius),
    fill='black'
)

# 保存为 PNG 文件
img.save('2,8.png')
print("图片已保存为 5.png")
