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
import matplotlib.patches as patches
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
func: calculate_trapz
input:x_data[batch,100,1] ,y_data [batch,100,2]
ouput:scalar
meaning: calculate the dircreate integral of the system
return 
'''

def calculate_trapz(x_data:torch.tensor,y_data:torch.tensor):
    '''
    calculate the trapezoid value of the system
    '''

    x_numpy=x_data.cpu().detach().numpy()
    y_numpy=y_data.cpu().detach().numpy()
    #y_numpy=[batch,100,1]
    trap=np.trapz(y_numpy,x_numpy,axis=1)
    trap=torch.from_numpy(trap).to(device="cuda")
    return trap

'''
func:calculate the fft of the data
input:real_data:torch.tensor and save the main freq nums
output:freq of the data like [batch,save_main_numbers]
'''

def get_top_frequencies_magnitudes_phases(signal, top_k=3, sampling_rate=50, device="cuda"):
    # Compute FFT
    f_transform = torch.fft.fft(signal).to(device)
    # Get positive frequencies
    num_samples = signal.size(0)
    positive_freqs = torch.fft.fftfreq(num_samples, d=1 / sampling_rate)[:num_samples // 2].to(device)

    # Get indices of top K magnitudes
    magnitudes = torch.abs(f_transform[:num_samples // 2])
    phases = torch.angle(f_transform[:num_samples // 2])
    _, top_indices = torch.topk(magnitudes, top_k)

    return positive_freqs[top_indices], magnitudes[top_indices], phases[top_indices]
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
                              variable_t,
                              in_fake_data:torch.Tensor,
                              in_real_data:torch.Tensor,
                              basis_str,left_matrix:torch.Tensor,coeffs:torch.Tensor,solus_str,
                            ):
    '''
    :param plot_path: str
    :param writer:tensorboard
    :param variable_t:[batch,100,1]
    :param in_fake_data:[batch,100,2]
    :param in_real_data:[batch,100,2]
    :param basis_str:[batch*2] because 2 z1 and z2
    :param basis_matrix: [batch,100,basis_num,2]
    :param coeffs:[batch,6]
    :param solus_str:a list that concludes batch*{'z1_solu': 'z1=sin(19*t) + cos(19*t)', 'z2_solu': 'z2=-sin(19*t) + cos(19*t)'}

    '''
    #record count_critic_step
    global count_critic_step
    count_critic_step+=1
    #get the batch size
    batch_size,_,_=in_fake_data.size()
    basis_num=left_matrix.shape[2]
    #length of solus_str is batch_size
    assert len(solus_str)==batch_size

    # data process
    in_fake_data=in_fake_data.cpu().detach()

    in_real_data=in_real_data.cpu().detach()
    #in_real_data=in_real_data.reshape(batch_size,100,2)
    variable_t=variable_t[0,:,0].cpu().detach().numpy() #[100,1]
    coeffs=coeffs.cpu().detach().numpy() #[batch,basis_num*2] first are z1_t

    basis_matrix=left_matrix.detach()#[batch,100,basis_num,2]

    #plot the sub figure

    plt.figure(figsize=(30, 30))
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
        #note because of the multi-gpu training
        #basis_matrix [batch,100,basis_num,2]
        ax0.plot(variable_t, basis_matrix[0,:,:,0].cpu().numpy(),linewidth=5,
                 label=basis_str[0][j])
        #fft basis_matrix
        basis_freqs, norm_spec,_ = get_top_frequencies_magnitudes_phases(basis_matrix[0:100,j],top_k=50,device="cpu")
        ax1.scatter(basis_freqs.numpy(),norm_spec.numpy(),s=1000-100*j,alpha=0.5,marker='*',
                    label=basis_str[0][j])
        #label the max
        max_index=torch.argmax(norm_spec)
        ax1.vlines(basis_freqs[max_index],0,norm_spec[max_index],colors='r',linestyles='dashed',linewidth=5)


    #plot the coeffs -violin
    z1_t_columns = [f'z1 coef {i + 1}' for i in range(basis_num)]
    z2_t_columns = [f'z2 coef {i + 1}' for i in range(basis_num)]
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

    for i in range(1):
        ax3.plot(variable_t, in_fake_data[i,:,0].numpy(),color='b',linestyle='--',
                 linewidth=2,label='z1fake_data_critic'+str(i))
        ax3.plot(variable_t, in_real_data[i,:,0].numpy(),color='k',
                 linewidth=2,label='z1real_data'+str(i))
        ax3.plot(variable_t, in_fake_data[i,:,1].numpy(),color='g',linestyle='--',
                 linewidth=2,label='z2fake_data_critic'+str(i))
        ax3.plot(variable_t, in_real_data[i,:,1].numpy(),color='r',
                 linewidth=2,label='z2real_data'+str(i))

        # fft data
        basis_freqs, basis_norm_spec,_ = get_top_frequencies_magnitudes_phases(in_fake_data[i, :, 0], top_k=50, device="cpu")

        ax4.scatter(basis_freqs.numpy(), basis_norm_spec.numpy(), s=1000 - 100 * i, alpha=0.5,
                    marker='*',
                    label='z1_fake'+str(i)
                    )
        # label the fake-max
        max_index = torch.argmax(basis_norm_spec)
        ax4.vlines(basis_freqs[max_index], 0, basis_norm_spec[max_index], colors='b',
                   linestyles='-.', linewidth=5,alpha=0.5)

        # fft data
        basis_freqs, basis_norm_spec,_ = get_top_frequencies_magnitudes_phases(in_real_data[i, :, 0], top_k=50, device="cpu")

        # label the real-max
        max_index = torch.argmax(basis_norm_spec)
        ax4.vlines(basis_freqs[max_index], 0, basis_norm_spec[max_index], colors='g',
                   linestyles='--',
                   linewidth=7,alpha=0.5)

        ax4.scatter(basis_freqs.numpy(), basis_norm_spec.numpy(), s=1000 - 10 * i, alpha=0.5,
                    marker='*',
                    label='z1_real'+str(i))


        #phase portrait
        ax5.plot(in_fake_data[i,:,0].numpy(),in_fake_data[i,:,1].numpy(),color='b',linestyle='--',
                    linewidth=2,label='fake_phase_z1'+str(i))
        ax5.plot(in_real_data[i,:,0].numpy(),in_real_data[i,:,1].numpy(),color='k', linewidth=2,
                    label='real_phase_z1'+str(i))
        fake_z1_t=in_fake_data[i,:,0]
        fake_z2_t=in_fake_data[i,:,1]
        #quiver
        u = [x - y for x, y in zip(fake_z1_t[1:].numpy(), fake_z1_t[:-1].numpy())]
        v = [x - y for x, y in zip(fake_z2_t[1:].numpy(), fake_z2_t[:-1].numpy())]
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
    path_analyze = plot_path + "/analyze_critic " + str(count_critic_step) + ".png"
    plt.savefig(path_analyze)
    plt.close()
    #
    # #--------------ridar
    #list basis_str to str
    #now is proper multiplication the basis form like : a1*basis +a2*basis
    #label for all the batch data
    # set figure
    fig, ax = plt.subplots(batch_size,1,figsize=(30, 30),
                           subplot_kw={'projection': 'polar'})
    plt.style.use('ggplot')

    batch_fake_symbolic={"z1_fake_symbolic_str":[],"z2_fake_symbolic_str":[]}
    batch_real_symbolic={"z1_real_symbolic_str":[],"z2_real_symbolic_str":[]}

    for i in range(batch_size):
        #pick the batch data
        ax[i].patch.set_edgecolor('black')
        ax[i].patch.set_linewidth(3)

        z1_t_coeff= coeffs[i,0:basis_num]
        z2_t_coeff= coeffs[i,basis_num:]

        #ridar basis structure:we just write the first batch data
        fake_labels = basis_str[0]
        print("label_len",len(fake_labels))

        #fake data zip for the basis and coeff
        batch_fake_symbolic['z1_fake_symbolic_str'] = ' + '.join(f'{s}*{c}' for s, c in zip(z1_t_coeff, fake_labels))
        batch_fake_symbolic['z2_fake_symbolic_str'] = ' + '.join(f'{s}*{c}' for s, c in zip(z2_t_coeff, fake_labels))

        #real data zip for the basis and coeffs
        batch_real_symbolic['z1_real_symbolic_str'] =  solus_str[i]['z1_solu']
        batch_real_symbolic['z2_real_symbolic_str'] =  solus_str[i]['z2_solu']

        print("batch_real_symbolic",batch_real_symbolic)


        #fake and real data in j is the batch
        # symlog
        ax[i].set_rscale('symlog', linthresh=1e-4)
        # polar
        ax[i].plot(np.radians(np.linspace(0, 360, len(fake_labels),
                                endpoint=False)),
                z1_t_coeff, marker='*',label='z1_fake__symbolic_str:'+ batch_fake_symbolic['z1_fake_symbolic_str'] )

        ax[i].plot(np.radians(np.linspace(0, 360, len(fake_labels),
                                endpoint=False)),
                z2_t_coeff, marker='*',label='z2_fake__symbolic_str:'+ batch_fake_symbolic['z2_fake_symbolic_str'] )
        # set tick
        ax[i].tick_params(axis='both', labelsize=6)
        ax[i].legend(fontsize=20,loc ='upper right')
        # set labels
        ax[i].set_xticks(np.radians(np.linspace(0, 360, len(fake_labels),
                                       endpoint=False)))
        ax[i].set_xticklabels(basis_str[0], fontsize=12, color='black')
        # spine
        ax[i].spines['polar'].set_visible(True)
        ax[i].spines['polar'].set_color('black')
        ax[i].spines['polar'].set_linewidth(1)
        # grid
        #title
        ax[i].set_title("fake data_compare_real_symbol:"+f"{batch_real_symbolic['z1_real_symbolic_str']}"+"_"\
                           +f"{batch_real_symbolic['z2_real_symbolic_str']}"
                           ,fontsize=title_size/2)


    path_ridar= plot_path + "/ridar_critic " + str(count_critic_step) + ".png"
    plt.savefig(path_ridar)
    plt.close()

    #save the ridar image to tensorboard
    image_ridar = Image.open(path_ridar)
    image_ridar_tensor = ToTensor()(image_ridar)
    writer.add_image('image_ridar', image_ridar_tensor,global_step=count_critic_step)
    #save analyze image to tensorboard
    image_analyze = Image.open(path_analyze)
    image_analyze_tensor = ToTensor()(image_analyze)
    writer.add_image('image_analyze', image_analyze_tensor,global_step=count_critic_step)


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
    print("mp4 save")
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
def calculate_fid_score(fake,real):
    '''

    :param fake: [100,2]
    :param real: [100,2]
    :return: [1]
    '''
    #
    mu_real = torch.mean(real, dim=0)
    mu_gen = torch.mean(fake, dim=0)
    cov_real = torch.cov(real.T)
    cov_gen = torch.cov(fake.T)
    # FID(x, g) = ||μ_x - μ_g||^2 + Tr(Σ_x + Σ_g - 2(Σ_xΣ_g)^{0.5})
    #svd to keep the matrix positive
    U, S, V = torch.linalg.svd(cov_real)
    sqrt_cov_real = U @ torch.diag(torch.sqrt(S)) @ V.T

    U, S, V = torch.linalg.svd(cov_gen)
    sqrt_cov_gen = U @ torch.diag(torch.sqrt(S)) @ V.T
    fid = torch.linalg.norm(mu_real - mu_gen) + torch.trace(cov_real + cov_gen - 2 *sqrt_cov_real @ sqrt_cov_gen)
    print("fid",fid)
    return fid

import torch.nn as nn
def calculate_mse(fake,real):
    '''

    :param fake: [100,2]
    :param real: [100,2]
    :return: [1]
    '''
    loss=nn.MSELoss()
    mse_loss=loss(fake,real)
    return mse_loss

'''
func:according th label to read the real data str
like z1:sint*cost 
like z2:cost*e^t 
return a dict of str
'''
def read_real_str(label:torch.tensor):

    real_str_list=[]
    for i in range(len(label)):
        #read the data colunmn=1
        value=pd.read_csv("../ode_dataset/easy_center/data"+str(label[i].item())+".csv",nrows=1)
        z1_solu=value['sol_z1'].values
        z2_solu=value['sol_z2'].values

        # eval
        dict1 = eval(z1_solu[0]+ "}")
        dict2 = eval("{" + z2_solu[0] )
        str_dict={**dict1,**dict2}
        real_str_list.append(str_dict)

    #like:
    # [{'z1_solu': 'z1=sin(12*t) + cos(12*t)', 'z2_solu': 'z2=-sin(12*t) + cos(12*t)'}]
    return real_str_list


'''
func:calcualte the diff of the data
input:real_data:torch.tensor and type
output:diff_grads:torch.tensor
'''
def calculate_diff_grads(real_data:torch.tensor,data_t:torch.tensor,type="center_diff")->torch.tensor:
    #real_data [batch,100,2]
    #data_t [batch,100,1]

    delta_t=data_t[:,1,0]-data_t[:,0,0]
    derivatives=0

    if type=="center_diff":

        central_diff = (real_data[:, 2:, :] - real_data[:, :-2, :]) / (2 * delta_t)
        start_diff = (real_data[:, 1, :] - real_data[:, 0, :]) / delta_t
        end_diff = (real_data[:, -1, :] - real_data[:, -2, :]) / delta_t
        # cat the start and end
        derivatives = torch.cat([start_diff.unsqueeze(1), central_diff, end_diff.unsqueeze(1)], dim=1)

    elif type=="chebyshev_diff":
        pass


    return derivatives



def calculate_fft(data,save_main_numbers=1):
    #data:[batch,100,2]
    #return:main freq:[batch,save_main_numbers,2]
    #2s-100, the sampling_rate=50hz
    # the resolution is 1/2s=0.02hz
    # the max freq is 25hz
    # the min freq is 0
    batch_size=data.shape[0]
    #freq_result:[batch,save_main_numbers,2]
    fft_result=torch.zeros((batch_size,save_main_numbers,2),
                           dtype=torch.float32,device="cuda")
    for i in range(batch_size):

        z1_top,_,_=get_top_frequencies_magnitudes_phases(data[i,:,0],top_k=save_main_numbers,
                                                     sampling_rate=50)
        z2_top,_,_=get_top_frequencies_magnitudes_phases(data[i,:,1],top_k=save_main_numbers,
                                                     sampling_rate=50)
        fft_result[i,:,0]=z1_top
        fft_result[i,:,1]=z2_top

    return fft_result





if __name__ == "__main__":

    print("utils_func.py is loaded")


