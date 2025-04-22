from PIL import Image, ImageOps

# 打开原图
img = Image.open('/Users/wangmusi/Documents/GitHub/Simulation/USAF/USAF_target_gray.png')
w, h = img.size
target = w  # 10560

# 计算需要补的高度
pad_total = target - h
pad_top = pad_total // 2
pad_bottom = pad_total - pad_top

# 填充图像
img_padded = ImageOps.expand(
    img,
    border=(0, pad_top, 0, pad_bottom),  # (left, top, right, bottom)
    fill=255  # 黑色
)

# 保存填充后的图像
output_path = 'USAF/padded_image.png'
img_padded.save(output_path)

print(f"Saved padded image to {output_path}")  # (10560, 10560)
