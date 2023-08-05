
import sympy as sp
import torch
import torch.nn as nn
import numpy as np
from utils.wgan_data import *
 # 定义生成器“基的系数网络”, 输入是一维度的噪声，输出是[100,1]的解
class Generator(nn.Module):
    def __init__(self,config):

        super(Generator, self).__init__()
        hidden_1=config["g_neural_network_width"]


        self.model = nn.Sequential(
            nn.Linear(300+config["zdimension_Gap"],10),
            nn.LeakyReLU(),
            nn.Linear(10,hidden_1),
            nn.LeakyReLU(),
            nn.Linear(hidden_1, 2),
            nn.LeakyReLU()
        )
        print("***have_init")
    #kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight,a=0,mode='fan_in')
    energy_regurla=0#
    energy=torch.zeros(batch_size,1,device="cuda") #消耗的energy
    generate_data=torch.zeros(batch_size,100,device="cuda")
    math_matrix=[]
    def create_matrix(self,data_t): #矩阵function

        data_numpy=data_t.cpu().clone().numpy()
        #构建数据矩阵，用来作用于数学矩阵，每一个batch为一列，100为行
        data_matrix=sp.Matrix(data_numpy)

        print("data_numpy",data_numpy.shape)
        # 创建符号变量
        x = sp.symbols('x')
        # 构建运算矩阵
        self.math_matrix = sp.Matrix([x**i for i in range(4)])
        print("math_matrix",self.math_matrix)
        basis=[0,1,2,3]#基的系数 暂定 生成1，x^1,x^2,x^3

        b_t=data_numpy
        result=np.zeros((batch_size,4,100))
        print("b_t",b_t.shape)

        for i in range(4):
            result[:,i,:]=data_matrix[i,:].applyfunc(lambda elem:elem**basis[i])
        print("resultshape",result.shape)


        # 打印函数的符号表示
        f_symbolic = sp.pretty(self.math_matrix)
        return result[:,1,:],result[:,2,:],result[:,3,:]#返回三个列，表示生成的列 第一个列是1，后面就做了运算 [batch,1,100]

    def print_coeff(self):
        print('coeff is',self.coeff)
        return self.coeff

    def calculate_generate(self,coeff,data_t): #计算生成的函数data_t的维度是【batch,100,3】
        #print(coeff)
        # 创建符号变量
        #构建100*1的numpy
        num_matirx=np.ones((batch_size,4,100))
        num_matirx[:,1,:],num_matirx[:,2,:],num_matirx[:,3,:]=self.create_matrix(data_t)
        #print(num_matirx.shape)
        num_matirx_tensor=num_matirx.astype(np.float32)
        num_matirx_tensor=torch.from_numpy(num_matirx).to(device="cuda").float()
        print("num_matirx_tensor",num_matirx_tensor.shape)
        print("coeff",coeff.shape)
        print("data_t",data_t.shape)
        #四列加起来成为一列
        for i in range(batch_size):
            a=coeff[i,0]*data_t[i,:] \
                      +coeff[i,1]*num_matirx_tensor[i,2,:]\
                      # +0*coeff[0,2]*num_matirx_tensor[2,:,:]\
                      # +0*coeff[0,3]*num_matirx_tensor[3,:,:]
            self.generate_data[i,:]=a
        #data_t用二范数
        # 计算一列数据的二范数 作为能量
            self.energy[i,:]=0.1*(coeff[i,0]**2)*torch.norm(data_t[i,:])\
               +(coeff[i,1]**2)*torch.norm(num_matirx_tensor[i,2,:])\
               #+torch.norm(num_matirx_tensor[2,:,:])\
               #+torch.norm(num_matirx_tensor[3,:,:])
        print("generate_data", self.generate_data.shape)
        print("energy", self.energy)
        return self.generate_data

    def forward(self, x,data_t):
        self.coeff=self.model(x)
        data=self.calculate_generate(self.coeff,data_t)
        return data

# 定义判别器网络 输入是【ini_con,t,ydot,y】，输出是一个数-打分
#[100,4]
class Discriminator(nn.Module):
    def __init__(self,config):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        #  #kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight,a=0,mode='fan_in')

    def forward(self, x):
        print(x.shape)
        critic=self.model(x)
        return critic
def wgan_model_save(generator:nn.Module,discriminator:nn.Module,save_path:str):

    torch.save(generator.state_dict(), save_path+'/generator.pth')
    torch.save(discriminator.state_dict(), save_path+'/discriminator.pth')
