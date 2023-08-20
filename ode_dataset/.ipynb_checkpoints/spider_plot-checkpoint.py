import pygal

# 调用Radar这个类，并设置雷达图的填充，及数据范围
radar_chart = pygal.Radar(fill = True, range=(0,5))
# 添加雷达图的标题
radar_chart.title = '活动前后员工状态表现'
# 添加雷达图各顶点的含义
radar_chart.x_labels = ['个人能力','QC知识','解决问题能力','服务质量意识','团队精神']

# 绘制两条雷达图区域
radar_chart.add('活动前', [3.2,2.1,3.5,2.8,3])
radar_chart.add('活动后', [4,4.1,4.5,4,4.1])

# 展示图像
radar_chart.render_to_file('radar_chart.svg')
