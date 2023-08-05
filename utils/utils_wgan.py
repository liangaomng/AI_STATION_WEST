import torch
from torch.utils.data import DataLoader, Dataset,TensorDataset
#关掉warning
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
from PIL import Image
from torchvision.transforms import ToTensor
import sympy as sp
import os

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)
    random.seed(seed)

# 定义全局变量用于记录函数被调用的次数
count_critic_step = 0
count_generator_step = 0

def plot_critic_tensor_change(train_init,writer,loss,variable_t,in_fake_data:torch.Tensor,in_real_data:torch.Tensor):

    global count_critic_step
    batch_size=in_fake_data.shape[0]
    count_critic_step+=1
    print("batch_size**",batch_size)
    # 开启交互模式
    plt.ion()
    plt.style.use('ggplot')  # 设置绘图风格
    in_fake_data=in_fake_data.cpu().detach().numpy()
    in_fake_data=in_fake_data.reshape(batch_size,100)
    in_real_data=in_real_data.cpu().detach().numpy()
    in_real_data=in_real_data.reshape(batch_size,100)
    variable_t=variable_t.cpu().detach().numpy()
    plt.figure(1)
    loss=loss.item()
    variable_t=variable_t.reshape(batch_size,100)
    for _ in range(batch_size):
        #看一个维度上最大和最小值 我们随机抽取batch 里面的第一个
        plt.scatter(x=variable_t[0,:],y=in_fake_data[_,:],c='r',
                    label='t-fake_data_critic',
                    marker='*',
                    legend='')
        plt.scatter(x=variable_t[0,:],y=in_real_data[_,:],c='b',
                    label='t-real_data'
                    ,marker='o')
    plt.legend()

    # 保存图形，并设置标题loss

    path=train_init.save_path +"/critic "+str(count_critic_step)+".png"
    plt.title("critic_loss:"+str(loss))
    plt.savefig(path)
    #legend 图例
    plt.close()  # 清除窗口
    image = Image.open(path)
    image_tensor = ToTensor()(image)
    writer.add_image('critic_Image', image_tensor,global_step=count_critic_step)

def plot_generator_tensor_change(train_init,writer,loss,variable_t,in_fake_data:torch.Tensor,in_real_data:torch.Tensor,conditional_data:torch.Tensor):
    global count_generator_step
    count_generator_step+=1
    batch_size=in_fake_data.shape[0]
    # 开启交互模式
    plt.ion()
    in_fake_data = in_fake_data.cpu().detach().numpy()
    in_fake_data = in_fake_data.reshape(batch_size, 100)
    in_real_data = in_real_data.cpu().detach().numpy()
    in_real_data = in_real_data.reshape(batch_size, 100)
    variable_t = variable_t.cpu().detach().numpy()
    plt.figure(1)
    #图片大小设定
    plt.figure(figsize=(30, 20))
    plt.style.use('ggplot')  # 设置绘图风格

    loss = loss.item()
    variable_t = variable_t.reshape(batch_size, 100)
    # 看一个维度上最大和最小值 我们随机抽取batch 里面的第一个
    for _ in range(batch_size):
        print(str(train_init.generator.print_coeff()[_,:]))
        print(train_init.generator.math_matrix[2])

        plt.scatter(x=variable_t[0, :], y=in_fake_data[_, :] ,
                    c='r',
                    label='t-fake_data_'+\
                          'y='+str(train_init.generator.print_coeff()[_,0].item())+"*"+str(train_init.generator.math_matrix[1])\
                           +str(train_init.generator.print_coeff()[_,1].item())+"*"+str(train_init.generator.math_matrix[2]),
                    marker="*",
                   )

        plt.scatter(x=variable_t[0, :], y=in_real_data[_, :],
                    c='b',label='t-real_data_'+'y',
                    marker="o",
                   )
    # 显示 legend，并设置位置为左上角，列数为 2，大小自适应
    plt.legend(loc='upper left', bbox_to_anchor=None)
    # 保存图形，并设置标题loss
    path=train_init.save_path+"/generator "+str(count_generator_step)+".png"
    plt.title("generator_loss:"+str(loss))
    plt.savefig(path)
    plt.close()  # 清除窗口
    image = Image.open(path)
    image_tensor = ToTensor()(image)

    writer.add_image('generartor_Image', image_tensor,global_step=count_generator_step)

#把crtic的png汇总起来，变成一个mp4
#把generator的png汇总起来,变成一个mp4
import cv2
import os
from natsort import natsorted
def images_to_video(search_path, search_name,output_filename, fps=1000):

    image_folder = search_path
    video_path = search_path+"/"+output_filename

    images = [img for img in os.listdir(image_folder) \
    if img.endswith(".png") and img.startswith(search_name)]
    images = natsorted(images)
    #把images里面的图片变成视频
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    print("mp4保存完成")


if __name__ == "__main__":
    print("utils_wgan.py is loaded")



