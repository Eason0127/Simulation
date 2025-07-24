import matplotlib.pyplot as plt

# 生成 z1 和对应的分辨率数据
start, end, step = 18.2, 24, 0.2
z1 = [round(start + i * step, 10) for i in range(int((end - start) / step) + 1)]
resolution = [27.8] * len(z1)

# 创建画布
fig, ax = plt.subplots(figsize=(10, 4))

# 绘制黄色水平线
ax.plot(
    z1,
    resolution,
    color='#EAC435',      # 黄色
    linewidth=2,
    alpha=0.8,
    label='R = 27.8 µm'
)

# 在每个点上用红色“×”标注
ax.scatter(
    z1,
    resolution,
    marker='x',
    color='red',
    label='Data points'
)

# 设置坐标标签、标题、刻度和网格
ax.set_xlabel("z1 (cm)", fontsize=14)
ax.set_ylabel("R (µm)", fontsize=14)
ax.set_title("Resolution R vs z1 (w = 700×5.86 µm)", fontsize=16)
ax.set_xticks([18, 19, 20, 21, 22, 23, 24])
ax.set_ylim(18, 30)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 添加图例
ax.legend()

plt.tight_layout()
plt.show()
