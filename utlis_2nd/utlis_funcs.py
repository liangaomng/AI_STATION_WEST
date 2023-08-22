import pandas as pd
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from natsort import natsorted
from torchvision.transforms import ToTensor
import argparse
import matplotlib.gridspec as gridspec
import ode_dataset.fft as please_fft
import math
import seaborn as sns
'''
this file contains all the functions that are used in the main file
func1:set_seed(seed)
'''
def set_seed(seed=42):
    '''
    set_seed
    '''

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)
    random.seed(seed)

'''
varibles:   golbal varibles to record critic
            and generator trainingstep
'''
count_critic_step = 0
count_generator_step = 0

'''
func:plot_critic_tensor_change()
meaning:plot function for critic 
which means that we plot the batch size data in the same figure
and basis function' shape
concludes:fake data and real data  and the real loss 
'''
def plot_critic_tensor_change(plot_path,writer,
                              loss,variable_t,
                              in_fake_data:torch.Tensor,
                              in_real_data:torch.Tensor,
                              basis_str,basis_matrix,coeffs:torch.Tensor,
                              energy:torch.Tensor):

    global count_critic_step
    batch_size=in_fake_data.shape[0]
    count_critic_step+=1
    basis_num=basis_matrix.shape[2]
    print("batch_size**",batch_size)

    # data process
    in_fake_data=in_fake_data.cpu().detach().numpy()
    in_fake_data=in_fake_data.reshape(batch_size,100,2)
    in_real_data=in_real_data.cpu().detach().numpy()
    in_real_data=in_real_data.reshape(batch_size,100,2)
    variable_t=variable_t[0,:,0].cpu().detach().numpy() #[100,1]
    coeffs=coeffs.cpu().detach().numpy() #[batch,8] first 4 are z1_t
    coeffs = coeffs.reshape(batch_size, 8)
    print("coeffs.shape",coeffs.shape)

    #plot the sub figure
    fig,ax= plt.subplots(3,2,figsize=(70, 30))
    plt.style.use('ggplot')  # fig set

    #plot layout
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1])
    ax0 = plt.subplot(gs[0])
    ax1= plt.subplot(gs[1])
    ax2= plt.subplot(gs[2])
    ax3= plt.subplot(gs[3])
    ax4= plt.subplot(gs[4])
    ax5= plt.subplot(gs[5])
    title_size=50
    for j in range(basis_num):
        #plot the basis data
        ax0.plot(variable_t, basis_matrix[0,:,j],linewidth=5,
                 label=basis_str[j])
        #fft basis_matrix
        basis_freqs, basis_norm_spec = please_fft.help_fft(basis_matrix[0,:,j])
        basis_norm_spec = [0 if math.isnan(x) else x for x in basis_norm_spec]
        ax1.scatter(basis_freqs,basis_norm_spec,s=1000-100*j,alpha=0.5,marker='*',
                    label=basis_str[j])

    #plot the coeffs -violin
    z1_t_columns = [f'z1 coef {i + 1}' for i in range(4)]
    z2_t_columns = [f'z2 coef {i + 1}' for i in range(4)]
    #concat list
    group_list= z1_t_columns + z2_t_columns
    df = pd.DataFrame(coeffs, columns=group_list)
    # data process
    df_melted = df.melt(value_name="Values", var_name="Groups")
    sns.violinplot(x="Groups", y="Values", data=df_melted,
                   ax=ax2,width=1.2,linewidth=2)

    ax0.set_title("basis function",fontsize=title_size)
    ax1.set_title("basis function fft",fontsize=title_size)
    ax2.set_title("coeffs",fontsize=title_size)
    #
    ax0.tick_params(labelsize=title_size/2)
    ax1.tick_params( labelsize=title_size/2)
    ax2.tick_params( labelsize=title_size/2)

    ax0.legend(fontsize=title_size/2)
    ax1.legend(fontsize=title_size/2)

    for i in range(batch_size):
        ax3.plot(variable_t, in_fake_data[i,:,0],color='b',linestyle='--',
                 linewidth=2,label='z1fake_data_critic'+str(i))
        ax3.plot(variable_t, in_real_data[i,:,0],color='k',
                 linewidth=2,label='z1real_data'+str(i))
        ax3.plot(variable_t, in_fake_data[i,:,1],color='g',linestyle='--',
                 linewidth=2,label='z2fake_data_critic'+str(i))
        ax3.plot(variable_t, in_real_data[i,:,1],color='r',
                 linewidth=2,label='z2real_data'+str(i))

        # fft data
        basis_freqs, basis_norm_spec = please_fft.help_fft(in_fake_data[i, :, 0])
        basis_norm_spec = [0 if math.isnan(x) else x for x in basis_norm_spec]
        ax4.scatter(basis_freqs, basis_norm_spec, s=1000 - 100 * i, alpha=0.5,
                    marker='*',
                    label='z1 fake'+str(i)
                    )
        basis_freqs, basis_norm_spec = please_fft.help_fft(in_real_data[i, :, 0])
        basis_norm_spec = [0 if math.isnan(x) else x for x in basis_norm_spec]
        ax4.scatter(basis_freqs, basis_norm_spec, s=1000 - 10 * i, alpha=0.5,
                    marker='*',
                    label='z1 real'+str(i))

        #phase portrait
        ax5.plot(in_fake_data[i,:,0],in_fake_data[i,:,1],color='b',linestyle='--',
                    linewidth=2,label='fake_phase'+str(i))
        ax5.plot(in_real_data[i,:,0],in_real_data[i,:,1],color='k', linewidth=2,
                    label='real_phase'+str(i))
        fake_z1_t=in_fake_data[i,:,0]
        fake_z2_t=in_fake_data[i,:,1]
        #quiver
        u = [x - y for x, y in zip(fake_z1_t[1:], fake_z1_t[:-1])]
        v = [x - y for x, y in zip(fake_z2_t[1:], fake_z2_t[:-1])]
        ax5.quiver(fake_z1_t[1:],fake_z2_t[1:],
                    u,v,
                  scale_units='xy', angles='xy', scale=0.3,color='r')
        real_z1_t=in_real_data[i,:,0]
        real_z2_t=in_real_data[i,:,1]
        u = [x - y for x, y in zip(real_z1_t[1:], real_z1_t[:-1])]
        v = [x - y for x, y in zip(real_z2_t[1:], real_z2_t[:-1])]
        ax5.quiver(real_z1_t[1:],real_z2_t[1:],
                    u,v,
                    scale_units='xy', angles='xy', scale=0.3,color='g')





    ax3.set_title("real and fake data",fontsize=title_size)
    ax3.tick_params(labelsize=title_size/2)
    ax3.legend(ncol=12,fontsize=title_size/6)

    ax4.set_title("real and fake data fft",fontsize=title_size)
    ax4.tick_params(labelsize=title_size/2)
    ax4.legend(ncol=10,fontsize=title_size/4)

    ax5.set_title("phase portrait",fontsize=title_size)
    ax5.tick_params(labelsize=title_size/2)
    ax5.legend(ncol=10,fontsize=title_size/4)

    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(1)
    plt.figure(figsize=(30, 20))
    plt.style.use('ggplot')  # fig set

    loss=loss.item()
    variable_t=variable_t.reshape(batch_size,100)
    for _ in range(batch_size):

        plt.scatter(x=variable_t[0,:],y=in_fake_data[_,:],c='r',
                    label='t-fake_data_critic',
                    marker='*',
                    legend='')
        plt.scatter(x=variable_t[0,:],y=in_real_data[_,:],c='b',
                    label='t-real_data'
                    ,marker='o')
    plt.legend()

    #save the figure

    path=plot_path+"/critic "+str(count_critic_step)+".png"
    plt.title("critic_loss:"+str(loss))
    plt.savefig(path)
    #legend
    plt.close()  # close
    #save the image to tensorboard
    image = Image.open(path)
    image_tensor = ToTensor()(image)
    writer.add_image('critic_Image', image_tensor,global_step=count_critic_step)

'''
func:plot_generator_tensor_change()
meaning:plot function for genertors 
which means that we plot the batch size data in the same figure
concludes:fake data and real data  and the real loss  
+conditional_data (freqs numbers and size)
未来想看随时间画的基的变化哈
'''
def plot_generator_tensor_change(train_init,writer,loss,variable_t,
                                 in_fake_data:torch.Tensor,
                                 in_real_data:torch.Tensor,
                                 conditional_data:torch.Tensor):
    global count_generator_step
    count_generator_step+=1
    batch_size=in_fake_data.shape[0]

    #data process
    in_fake_data = in_fake_data.cpu().detach().numpy()
    in_fake_data = in_fake_data.reshape(batch_size, 100)
    in_real_data = in_real_data.cpu().detach().numpy()
    in_real_data = in_real_data.reshape(batch_size, 100)
    variable_t = variable_t.cpu().detach().numpy()

    plt.figure(1)
    plt.figure(figsize=(30, 20))
    plt.style.use('ggplot')  # fig style

    loss = loss.item()
    variable_t = variable_t.reshape(batch_size, 100)
    # plot
    for _ in range(batch_size):

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
    # legnd
    plt.legend(loc='upper left', bbox_to_anchor=None)
    # title is generator_loss
    path=train_init.save_path+"/generator "+str(count_generator_step)+".png"
    plt.title("generator_loss:"+str(loss))
    plt.savefig(path)
    plt.close()  # 清除窗口
    image = Image.open(path)
    image_tensor = ToTensor()(image)

    writer.add_image('generartor_Image', image_tensor,global_step=count_generator_step)

'''
func:images_to_video()
meaning:search and save .mp4
'''
def images_to_video(search_path:str, search_name:str,
                    output_filename:str,
                    fps=1000):
    '''
    :param search_path:     a file which contains the png
    :param search_name:     like "generator" and "critic"
    :param output_filename:  output name
    :param fps: 1s

    '''

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
'''
func:Get_test_args
meaning:get argus for the tese
return a list
'''
def Get_test_args(filepath)->list:
    '''
    :return: a list of dict
    '''
    df=pd.read_csv(filepath,header=0)
    #name of columns is the keys of dict

    keys=df.columns.values.tolist()
    #values of columns  2nd row is the values of dict
    values=df.values.tolist()
    #dict
    argu_dict=[dict(zip(keys,values[i])) for i in range(len(values))]
    for i in range(len(argu_dict)):
        for key in argu_dict[i].keys():
            #del , and change to float
            if type(argu_dict[i][key])==str:
                data_values=argu_dict[i][key].split(",")
                data_values=[float(data_values[i]) for i in range(len(data_values))]
                argu_dict[i][key]=data_values
    #return dict
    return argu_dict

'''
func:argu_parse
meaning:parse the argu
return argus
'''
def argu_parse():
    parser = argparse.ArgumentParser(description="wgan_v1 Argument Parser")
    # Add arguments here
    parser.add_argument("--name",default="ode_neural_operator",type=str, help="ode")
    parser.add_argument("--generator_num",default=1, type=int, help="generator_num ")
    parser.add_argument("--discriminator_num",default=1, type=int, help="discriminator_num ")
    parser.add_argument("--batch_size",default=1, type=int, help="batch_size ")
    parser.add_argument("--num_epochs",default=100, type=int, help="num_epochs ")
    parser.add_argument("--noise_dim",default=1, type=int, help="gaussian_noise_dim ")
    parser.add_argument("--mean",default=0.0, type=float, help="gaussian_mean ")
    parser.add_argument("--stddev",default=0.01, type=float, help="gaussian_stddev ")
    parser.add_argument("--gen_neural_network_deep",default=2, type=int, help="generator_deep ")
    parser.add_argument("--discri_neural_network_deep", default=2, type=int,
                        help="discriminator_deep ")
    parser.add_argument("--seed", default=42, type=int, help="seed ")
    parser.add_argument("--g_neural_network_width", default=512, type=int,
                        help="gen_num_neurons ")
    parser.add_argument("--dis_neural_network_width", default=256, type=int,
                        help="dis_num_neurons ")
    parser.add_argument("--energy_penalty_argu", default=0, type=float,
                        help="energy_penalty_argu-beta")
    parser.add_argument("--argue_basis", default=[1,1], type=list, help="[basis_number"\
                        ",basis_type(1:x^*,2:cos(x))]")
    parser.add_argument("--g_learning_rate", default=1e-3, type=float, help="g_learning_rate")
    parser.add_argument("--d_learning_rate", default=1e-3, type=float, help="d_learning_rate")
    parser.add_argument("--lipschitz_clip", default=0.01, type=float,help="lipschitz_clip")
    parser.add_argument("--iter_generator", default=1, type=int, help="iter_generator")
    parser.add_argument("--activation_function",default=["Relu"],type=list,help="activation")
    parser.add_argument("--savepath",default="../tb_info/compare_multi_argus",type=str,help="model_savepath")
    parser.add_argument("--denote", default="..denote", type=str, help="denote")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print("utils_func.py is loaded")

