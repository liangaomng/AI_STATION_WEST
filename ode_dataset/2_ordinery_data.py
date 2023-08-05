import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from fft import *

# 定义符号变量和时间
t = sp.Symbol('t')
z1 = sp.Function('z1')(t)
z2 = sp.Function('z2')(t)

#作为数据集,这里的数据集是一个矩阵
#[a11,a12,a21,a22] 作为label
#a11的变化规律如下 -5到5 共11
#初始条件z1(0)是-5到5 共11
#初始条件z2(0)是-5到5 共11
#所以数据集相当于 11的6次方
#时间0-2s 100个点



def demo():
    # 定义输入矩阵A
    a_values = [  # 可以添加更多的矩阵
        [[1,-2], [2, 1]],  # A1 【行的值】
    ]
    # 定义状态空间模型
    A = sp.Matrix(a_values[0])
    state_space_model = A * sp.Matrix([z1, z2])
    title_size=30
    # 定义不同的初值条件
    initial_conditions = [ ]

    # 使用嵌套的循环生成两个变量的组合
    for i in range(0,11):      # 变量1从-5到5
        for j in range(0,11):
            print("i",i)
            z1_ini_values=i-5
            z2_ini_values=j-5
            initial_conditions.append({z1.subs(t, 0): z1_ini_values,
                                       z2.subs(t, 0): z2_ini_values})
    print("initial_conditions",initial_conditions)

    plt.style.use('ggplot')
    line_shape=['r','g','b','k']
    # 定义时间范围
    time_range = np.linspace(0, 2, 100)
    # 创建一个正方形图，包含三个子图
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    tick_color='k'
    tick_size=20
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
        axes[0].set_xlim(-5,5)
        axes[0].set_ylim(-10, 10)
        #加x 和y
        axes[0].plot([0,0],[-10,10],color='k',linestyle='--')
        axes[0].plot([-10,10],[0,0],color='k',linestyle='--')
        axes[0].plot(z1_values, z2_values,label=f'Initial Condition {i+1}',
                 alpha=0.5,
                 color=line_shape[i%4]
                 )
        # 将列表转换为集合，然后计算差集
        difference_z1 = [x - y for x, y in zip(z1_values[1:], z1_values[:-1])]
        difference_z2 = [x - y for x, y in zip(z2_values[1:], z2_values[:-1])]

        # #画出向量场
        axes[0].quiver(z1_values[:-1], z2_values[:-1],
                   difference_z1,
                   difference_z2,
                   scale_units='xy', angles='xy', scale=2, color='r')
        axes[0].set_xlabel('z1',fontsize=title_size)
        axes[0].set_ylabel('z2',fontsize=title_size)
        axes[0].set_title('phase space', fontsize=title_size)
        #axes[0].legend(loc='best',ncol=3,fontsize=title_size/2)

        axes[0].tick_params(axis='x', colors=tick_color)
        axes[0].tick_params(axis='y', colors=tick_color)

        # 设置横纵坐标的大小
        axes[0].tick_params(axis='x', labelsize=tick_size)
        axes[0].tick_params(axis='y', labelsize=tick_size)
        axes[0].grid(True)
        # 画出time空间轨迹图

        axes[1].set_xlim(0, 2)
        #画出轨迹 t 和values
        axes[1].plot(time_range,z1_values,label=f'Initial Condition {i+1}')
        axes[1].plot(time_range,z2_values,label=f'Initial Condition {i+1}')
        axes[1].set_xlabel('t',fontsize=title_size)
        axes[1].set_ylabel('z',fontsize=title_size)
        axes[1].set_title('time space',fontsize=title_size)
        axes[1].grid(True)

        # 画出频谱图
        z1_freqs,z1_norm_spec=help_fft(z1_values)
        z2_freqs,z2_norm_spec=help_fft(z2_values)

        print("z1 the highest freq",z1_freqs[np.argmax(z1_norm_spec)])
        print("z2 the highest freq",z2_freqs[np.argmax(z2_norm_spec)])
        axes[2].plot(z1_freqs,z1_norm_spec,label=f'Initial Condition {i+1}')
        axes[2].plot(z2_freqs,z2_norm_spec,label=f'Initial Condition {i+1}')
        axes[2].set_xlabel('Frequency (Hz)',fontsize=title_size)
        axes[2].set_ylabel('Normalized Amplitude',fontsize=title_size)
        axes[2].set_title('Normalized Frequency Spectrum',
                          fontsize=title_size)
        #axes[2].legend(loc='best',ncol=3,fontsize=title_size/2)
        axes[2].grid(True)
    plt.subplots_adjust(wspace=0.3)
    # 自动调整图的大小和字体大小
    fig.tight_layout()
    plt.savefig("phase_space.png")
    plt.show()
def generate_2order_data(A_matrix,z1_ini,z2_ini,time_range):
    # 定义输入矩阵A
    a_values = A_matrix
    # 定义状态空间模型
    A = sp.Matrix(a_values)
    state_space_model = A * sp.Matrix([z1, z2])
    title_size = 30
    # 定义不同的初值条件
    initial_conditions = []
    initial_conditions.append({z1.subs(t, 0): z1_ini,
                                z2.subs(t, 0): z2_ini})

    # # 解析微分方程并添加当前初值条件
    solutions = sp.dsolve([z1.diff(t) - state_space_model[0],
                           z2.diff(t) - state_space_model[1]],
                           ics=initial_conditions[0])

    # 获取z1和z2的解析表达式
    z1_solution = solutions[0].rhs
    z2_solution = solutions[1].rhs
    solu={"z1_solu":0,"z2_solu":0}
    solu["z1_solu"]="z1="+str(z1_solution)
    solu["z2_solu"]="z2="+str(z2_solution)
    # 用lambdify将解析表达式转换为可以进行数值计算的函数
    z1_func = sp.lambdify(t, z1_solution, modules=['numpy'])
    z2_func = sp.lambdify(t, z2_solution, modules=['numpy'])


    z1_values = []
    z2_values = []
    for j in range(len(time_range)):
        z1_values.append(z1_func(time_range[j]))
        z2_values.append(z2_func(time_range[j]))
    return  z1_values,z2_values,solu

def handle_matrix(a_values,time_range,ini):
    '''
    input:matrix/time_range[100,1]/ini
    output:[100,9]+解z1和z2的表达式
    '''

    z1, z2,solu = generate_2order_data(a_values,
                                  z1_ini=ini[0][0],
                                  z2_ini=ini[0][1],
                                  time_range=time_range)
    # condition重复成100*6
    condition = np.concatenate((ini, a_values), axis=0)
    condition = condition.reshape(1, 6)
    #把solu重复
    solu = np.repeat(solu, 100, axis=0)
    solu = solu.reshape(100, 1)
    print(solu)

    ic_condition = np.repeat(condition, 100, axis=0)
    time_range = np.expand_dims(time_range, axis=0)
    time_range = time_range.transpose()
    data = np.concatenate((ic_condition,time_range), axis=1)
    # 使用np.expand_dims函数增加一个维度
    z1 = np.expand_dims(z1, axis=0)
    z2 = np.expand_dims(z2, axis=0)
    # 将z1和z2合并为一个数组
    z = np.concatenate((z1, z2), axis=0)
    z = z.transpose()
    # 将condition和data合并为一个数组
    data = np.concatenate((data, z), axis=1)
    # 将solu和data合并为一个数组
    data = np.concatenate((data, solu), axis=1)
    return data
def visual_data():
    '''
    提供相位图，时间图，频谱图
    '''
import torch
if __name__ == '__main__':

    # 定义输入矩阵A
    a_values = [[1, 1],[4,-2]]# 按照行的模式
    ini=[[1,0]]
    # 定义时间范围
    time_range = np.linspace(0, 2, 100)
    #1*100*9的数据维度
    result=np.zeros((1,100,10))
    # 制作数据集
    for i in range(1):
        data=handle_matrix(a_values,time_range,ini)
        new_result = np.expand_dims(data, axis=0)
        result = np.concatenate((result, new_result), axis=0)
    #消除第一个全零的维度
    result=result[1:,:,:]
    #tensor=result[:,:,0:].astype(np.float32)
    #保存csv
    np.savetxt('data.csv',result[0],delimiter=',',fmt='%s')
    exit()
    #保存tesnor
    tensor=torch.from_numpy(tensor)
    print(result.shape)
    # 画出相空间轨迹图





