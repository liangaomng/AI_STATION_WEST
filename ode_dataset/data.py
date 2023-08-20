import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# 定义符号变量和时间
t = sp.Symbol('t')
z1 = sp.Function('z1')(t)
z2 = sp.Function('z2')(t)

# 定义输入矩阵A
a_values = [  # 可以添加更多的矩阵
    [[-1,0], [0, -1]],  # A1
    [[-1, 0], [0, -2]],  # A2
]

# 定义状态空间模型
A = sp.Matrix(a_values[0])
state_space_model = A * sp.Matrix([z1, z2])

# 定义不同的初值条件
initial_conditions = [

]
# 计算z1和z2在给定时间范围内的数值

# 使用嵌套的循环生成两个变量的组合
for i in range(0,11):      # 变量1从-5到5
    for j in range(0,11):
        print("i",i)
        z1_ini_values=i-5
        z2_ini_values=j-5
        initial_conditions.append({z1.subs(t, 0): z1_ini_values,
                                   z2.subs(t, 0): z2_ini_values})
print("initial_conditions",initial_conditions)
# 创建图形窗口
plt.figure(figsize=(20, 20))
plt.style.use('ggplot')
line_shape=['r','g','b','k']
# 定义时间范围
time_range = np.linspace(0, 9, 10)

# 循环绘制不同初值条件下的相空间轨迹
for i, _ in enumerate(initial_conditions):

    ics=initial_conditions[i]
    #print("i====",ics)
    # 解析微分方程并添加当前初值条件
    solutions = sp.dsolve([z1.diff(t) - state_space_model[0],
                           z2.diff(t) - state_space_model[1]],
                           ics=ics)

    # 获取z1和z2的解析表达式
    z1_solution = solutions[0].rhs
    z2_solution = solutions[1].rhs

    print(z1_solution)

    # 用lambdify将解析表达式转换为可以进行数值计算的函数
    z1_func = sp.lambdify(t, z1_solution, modules=['numpy'],)
    z2_func = sp.lambdify(t, z2_solution, modules=['numpy'])

    z1_values=[]
    z2_values=[]
    for j in range(len(time_range)):
        z1_values.append(z1_func(time_range[j]))
        z2_values.append(z2_func(time_range[j]))
    print(z1_values)
    print(z2_values)

    # 画出相空间轨迹图
    plt.subplot(1,1,1)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    #加x 和y
    plt.plot([0,0],[-10,10],color='k',linestyle='--')
    plt.plot([-10,10],[0,0],color='k',linestyle='--')
    plt.plot(z1_values, z2_values,label=f'Initial Condition {i+1}',
             alpha=0.5,
             color=line_shape[i%4]
             )
    # 将列表转换为集合，然后计算差集
    difference_z1 = [x - y for x, y in zip(z1_values[1:], z1_values[:-1])]
    difference_z2 = [x - y for x, y in zip(z2_values[1:], z2_values[:-1])]

    # #画出向量场
    plt.quiver(z1_values[:-1], z2_values[:-1],
               difference_z1,
               difference_z2,
               scale_units='xy', angles='xy', scale=6, color='r')

# 添加图例和标签
plt.xlabel('z1(t)')
plt.ylabel('z2(t)')
plt.title('phase space'+' saddle')
plt.legend(loc='best',ncol=3)
plt.grid(True)
plt.show()