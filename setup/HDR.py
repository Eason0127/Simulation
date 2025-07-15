import cv2
import numpy as np
import os
import imageio.v2 as imageio
# -----------------------------------------------------------------------------
# 1. 参数设置
# -----------------------------------------------------------------------------
base_dir       = r"C:\Users\GOG\Desktop\HDR"
exp_times_ms   = [9.98, 20.02, 29.98, 40.02, 49.99, 60.03, 69.99]
dark_frame_cnt = 5

# -----------------------------------------------------------------------------
# 2. 生成 Master Dark（灰度黑电流校正）
# -----------------------------------------------------------------------------
dark_master = {}
for t in exp_times_ms:
    ds = []
    for i in range(1, dark_frame_cnt+1):
        fn = os.path.join(base_dir, f"dark_{t:.2f}_{i}.png")
        d  = cv2.imread(fn, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if d.dtype == np.uint16:
            d = d.astype(np.float32) / 65535.0
        else:
            d = d.astype(np.float32) / 255.0
        ds.append(d)
    dark_master[t] = np.mean(ds, axis=0)

# -----------------------------------------------------------------------------
# 3. 读取曝光图、减暗场、归一化到 [0,1]
# -----------------------------------------------------------------------------
imgs_gray = []
times_sec  = np.array(exp_times_ms, dtype=np.float32) / 1000.0

for t in exp_times_ms:
    fn = os.path.join(base_dir, f"exp_{t:.2f}ms.png")
    im = cv2.imread(fn, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    else:
        im = im.astype(np.float32) / 255.0
    im = np.clip(im - dark_master[t], 0.0, 1.0)
    imgs_gray.append(im)

# -----------------------------------------------------------------------------
# 4. 对齐：单通道→3通道→对齐→单通道
# -----------------------------------------------------------------------------
# 转 8‑bit
imgs8     = [(i*255).clip(0,255).astype(np.uint8) for i in imgs_gray]
# 扩为 BGR 三通道
imgs_align= [cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in imgs8]

alignMTB = cv2.createAlignMTB()
alignMTB.process(imgs_align, imgs_align)

# 降回灰度 & 归一化
imgs8_aligned = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs_align]
imgs_aligned  = [i.astype(np.float32)/255.0 for i in imgs8_aligned]

# -----------------------------------------------------------------------------
# 5. HDR 合成 (Debeveс) & 保存 EXR
# -----------------------------------------------------------------------------
merge = cv2.createMergeDebevec()
hdr   = merge.process(imgs8_aligned, times_sec)  # 单通道 float32

# 假设 `hdr` 是你合成后的 float32 单通道 HDR 图
hdr_path = os.path.join(base_dir, "fixed_output.hdr")
# format='HDR-FI' 指定 Radiance HDR 插件
imageio.imwrite(hdr_path, hdr, format="HDR-FI")
print("✅ HDR 已保存 (Radiance .hdr)：", hdr_path)

# -----------------------------------------------------------------------------
# 6. 色调映射 → 生成 LDR 预览
# -----------------------------------------------------------------------------
# 6.1 先把单通道 hdr 扩为 BGR 三通道
hdr_bgr = cv2.cvtColor(hdr, cv2.COLOR_GRAY2BGR)

# 6.2 Reinhard 色调映射
tonemap = cv2.createTonemapReinhard(gamma=2.2)
ldr_bgr = tonemap.process(hdr_bgr)

# 6.3 转为灰度 & 8‑bit
ldr_gray = cv2.cvtColor(ldr_bgr, cv2.COLOR_BGR2GRAY)
ldr8     = np.clip(ldr_gray * 255, 0, 255).astype(np.uint8)

ldr_path = os.path.join(base_dir, "fixed_output_ldr.png")
cv2.imwrite(ldr_path, ldr8)
print(f"✅ LDR 预览已保存：{ldr_path}")
