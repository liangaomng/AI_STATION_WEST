import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
plt.style.use('ggplot')

# 创建数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 创建一个正方形图，包含三个子图
fig, axes = plt.subplots(1, 3, figsize=(50, 50))

# 在第一个子图上进行绘图
axes[0].plot(x, y)
axes[0].set_aspect('equal')  # 设置子图的纵横比相等

# 添加第一个子图的局部放大效果
#边框变黑

axins = inset_axes(axes[0], width="30%", height="30%", loc=4)
axins.plot(x, y)
axins.set_xlim(2, 4)
axins.set_ylim(-0.5, 0.5)
# 修改局部放大图形的边框颜色为黑色
axins.spines['top'].set_color('black')
axins.spines['bottom'].set_color('black')
axins.spines['left'].set_color('black')
axins.spines['right'].set_color('black')
mark_inset(axes[0], axins, loc1=2, loc2=3, fc='black',ec="0.1")

# 在第二个和第三个子图上进行绘图
axes[1].scatter(x, y)
axes[1].set_aspect('equal')
axes[2].imshow(np.random.rand(10, 10))
axes[2].set_aspect('equal')

# 添加标题和标签
axes[0].set_title("Line Plot")
axes[1].set_title("Scatter Plot")
axes[2].set_title("Image Plot")
fig.suptitle("Three Subplots with a Zoomed-in Inset")

# 调整子图之间的间距
plt.subplots_adjust(wspace=0.3)

# 显示图形
plt.show()