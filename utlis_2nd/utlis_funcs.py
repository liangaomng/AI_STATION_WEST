import pandas as pd
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

import os
from natsort import natsorted
import argparse
import ot


import utlis_2nd.gan_nerual as gan_nerual
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
    torch.backends.cudnn.enabled = True
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
import matplotlib.colors as colors
def plot_critic_tensor_change(plot_path,writer,
                              variable_t,wgan_loss,div_fouier,div_gp,crtic_all_loss,
                              critic_numpy:np.ndarray,
                              now_epoch,epoch
                            ):
    '''
    :param plot_path: str
    :param writer:tensorboard
    :param variable_t:[batch,100,1]
    :param wgan_loss:scalar
    :param div_dp:scalar
    '''
    #record count_critic_step
    global count_critic_step
    count_critic_step+=1
    #get the batch size
    wgan_loss=wgan_loss.cpu().detach().numpy()
    div_gp=div_gp.cpu().detach().numpy()
    crtic_all_loss=crtic_all_loss.cpu().detach().numpy()
    div_fouier=div_fouier.cpu().detach().numpy()

    #prepare the plot

    #fill
    critic_numpy[0,now_epoch]=   wgan_loss #
    critic_numpy[1, now_epoch] = div_gp #
    critic_numpy[2, now_epoch] = div_fouier #
    critic_numpy[3,now_epoch] =  crtic_all_loss#

    #plot the sub figure
    plt.figure(figsize=(8, 6))
    plt.grid(True)
    #note that the first dimension y axis is z1 and z2
    norm = colors.SymLogNorm(linthresh=1e-3, linscale=1, vmin=-1e-2, vmax=1e+2)
    plt.pcolor(critic_numpy, cmap='RdBu', norm=norm)
    cbar=plt.colorbar()
    cbar.set_label('symLog Scale')

    plt.xlabel('Epoch', fontsize=22)
    # Set the y-axis ticks and labels to 1, 2 and rotate the labels
    y_positions = [0.5, 1.5, 2.5,3.5]
    y_labels = [r'$wgan_{loss}$', r'$div_{gp}$',r'$div_{fourier}$',r'$critic_{loss}$']
    plt.axhline(y=1, color='black', linewidth=2)
    plt.axhline(y=2, color='black', linewidth=2)
    plt.axhline(y=3, color='black', linewidth=2)
    plt.yticks(y_positions, y_labels, fontsize=22)

    # axis
    ax = plt.gca()

    #spines
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)  # 可以调整线的宽度

    # Set the y-axis tick parameters to hide the tick marks and set the tick label size
    plt.gca().yaxis.set_tick_params(size=0)
    plt.gca().tick_params(axis='y', labelsize=22)
    plt.gca().tick_params(axis='x', labelsize=22)

    plt.title('critic dynamic response', fontsize=22)
    plt.tight_layout()

    plt.savefig(plot_path + "/critic_loss.png")

    plt.close()

def plot_generator_tensor_change(plot_path,writer,
                              variable_t,gradient_loss,ini_loss,score_fake_out,generator_all_loss,
                              generator_numpy:np.ndarray,
                              now_epoch,epoch):

    #record count_critic_step
    global count_critic_step
    count_critic_step+=1
    #get the batch size
    gradient_loss=gradient_loss.cpu().detach().numpy()
    ini_loss=ini_loss.cpu().detach().numpy()
    score_fake_out=score_fake_out.cpu().detach().numpy()
    generator_loss=generator_all_loss.cpu().detach().numpy()

    #prepare the plot
    #fill
    generator_numpy[0,now_epoch]=   gradient_loss #
    generator_numpy[1, now_epoch] = ini_loss #
    generator_numpy[2,now_epoch] =  score_fake_out#
    generator_numpy[3,now_epoch] =  generator_all_loss#
    #plot the sub figure
    plt.figure(figsize=(8, 6))
    #note that the first dimension y axis is z1 and z2
    norm = colors.SymLogNorm(linthresh=1e-3, linscale=1, vmin=-1e-2, vmax=1e+2)
    plt.pcolor(generator_numpy, cmap='RdBu',norm=norm)
    cbar=plt.colorbar()
    cbar.set_label('Log Scale')

    plt.xlabel('Epoch', fontsize=22)
    # Set the y-axis ticks and labels to 1, 2 and rotate the labels
    y_positions = [0.5, 1.5, 2.5,3.5]
    y_labels = [r'$gradient_{loss}$', r'$ini_{loss}$',r'$score_{fake}$',r'$generator_{loss}$']
    plt.axhline(y=1, color='black', linewidth=2)
    plt.axhline(y=2, color='black', linewidth=2)
    plt.axhline(y=3, color='black', linewidth=2)
    plt.yticks(y_positions, y_labels, fontsize=22)
    # axis
    ax = plt.gca()

    # spines
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)  # 可以调整线的宽度

    # Set the y-axis tick parameters to hide the tick marks and set the tick label size
    plt.gca().yaxis.set_tick_params(size=0)
    plt.gca().tick_params(axis='y', labelsize=22)
    plt.gca().tick_params(axis='x', labelsize=22)

    plt.title("genertor dynamic response", fontsize=22)

    plt.tight_layout()
    plt.savefig(plot_path + "/generator_loss.png")
    plt.close()

def plot_freq_4_data_soft_argmax(plot_path,labels,data,name):
    '''
    in this function we plot the first batch data,note:we have reduced the average
    :param plot_path: str
    :param labels: str like "fake "
    :param data: tensor,[batch,100,2]
    :param fake_freq: tensor,[batch,2] because z1 and z2
    ：param name: str “/data1.png”
    '''
    data_t=np.linspace(0,2,100)
    #get the first batch data
    real_data=data[:,:,:]  #[batch,100,2]
    #average the data
    average_data=torch.mean(real_data,dim=1) #[1,2]
    real_data=real_data-average_data.unsqueeze(1)
    varible_num=data.shape[2]
    #get the real freq
    #please help me to plot the real and fake freq
    #plot the real freq and fake freq
    plt.figure(figsize=(20, 6))
    plt.grid(True)
    color = ['red', 'blue']
    line=['ro','bo']

    #real data
    plt.subplot(1, 3, 1)
    plt.grid(True)
    for i in range(varible_num):
        real=data[0,:,i].cpu().detach().numpy()
        plt.plot(data_t,real,label=labels+"_"+rf"$\mathbf {{Z_{i+1}}}$",color=color[i])
    plt.legend(loc='upper right')
    plt.title('Time Domain Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(labels+" time domain")

    #real freq
    plt.subplot(1, 3, 2)
    plt.grid(True)
    fs = 50#sampling frequency
    beta=1
    soft_argfreq=gan_nerual.compute_spectrum(real_data,beta) #return [1,2]
    soft_argfreq=soft_argfreq[0:1,:]/(2*torch.pi)
    resolution = 0.5
    #plot
    for i in range(varible_num):
        signal=real_data[0,:,i]
        fft_values = torch.fft.rfft(signal).abs() #[51]
        arr = np.linspace(0, fs*resolution, fs+1)  # 使用step=0.5
        lines=plt.stem(arr, fft_values.cpu().detach().numpy(),
                 label="Z_"+str(i+1),linefmt=line[i],use_line_collection=True)

        plt.setp(lines, color=color[i], alpha=0.7)
        plt.axvline(soft_argfreq[0,i].cpu().detach().numpy(), alpha=0.7,
                    color=color[i], linestyle='--',linewidth=3,
                    label='Soft_argmax with'+r"$T_"+str(beta)+"}$"+" for "\
                          +r"$ \mathbf {Z_"+str(i+1)+"}$")


    plt.title('Magnitude of FFT')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.title('Frequency Domain')

    plt.subplot(1, 3, 3)
    plt.grid(True)
    data_t = torch.from_numpy(data_t)
    data_t=data_t.reshape(1,100,1)
    diff_grads=calculate_diff_grads(real_data,data_t,type="center_diff",plt_show=True)


    plt.tight_layout()
    plt.savefig(plot_path + name)

    plt.show()
    plt.close()
    return diff_grads


def plot_conditions_4_real_fake(plot_path,real_condition,fake_condition):
    pass

'''
func:images_to_video()
meaning:search and save .mp4
# '''
# def images_to_video(search_path:str, search_name:str,
#                     output_filename:str,
#                     fps=1000):
#     '''
#     :param search_path:     a file which contains the png
#     :param search_name:     like "generator" and "critic"
#     :param output_filename:  output name
#     :param fps: 1s
#     '''
#
#     image_folder = search_path
#     video_path = search_path+"/"+output_filename
#
#     images = [img for img in os.listdir(image_folder) \
#
#     if img.endswith(".png") and img.startswith(search_name)]
#     images = natsorted(images)
#     #把images里面的图片变成视频
#     frame = cv2.imread(os.path.join(image_folder, images[0]))
#     height, width, layers = frame.shape
#     #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
#
#     for image in images:
#         video.write(cv2.imread(os.path.join(image_folder, image)))
#     print("mp4 save")
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
        value=pd.read_csv("../ode_dataset/complex_center_dataset/data"+str(label[i].item())+".csv",nrows=1)
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


def five_point_stencil(data,dt)->torch.tensor:
    #
    h=dt
    central_diff = (-data[:, 4:, :] + 8 * data[:, 3:-1, :] - 8 * data[:, 1:-3, :] + data[:, :-4, :]) \
                   / (12*h)

    start_diff_1 = (-3 * data[:, 0, :] + 4 * data[:, 1, :] - data[:, 2, :]) / (2*h)
    start_diff_2 = (-3 * data[:, 1, :] + 4 * data[:, 2, :] - data[:, 3, :]) / (2*h)

    end_diff_1 = (3 * data[:, -2, :] - 4 * data[:, -3, :] + data[:, -4, :]) / (2*h)
    end_diff_2 = (3 * data[:, -1, :] - 4 * data[:, -2, :] + data[:, -3, :]) / (2*h)


    # combine and return [batch,100,2]
    result = torch.cat([start_diff_1.unsqueeze(1),
                        start_diff_2.unsqueeze(1),
                        central_diff,
                        end_diff_1.unsqueeze(1),
                        end_diff_2.unsqueeze(1)], dim=1)

    return result

def chebyshev_appro(real_data,deg=10):

    batch_size, sequence_length, feature_dim = real_data.shape
    derivative = torch.zeros_like(real_data)
    for i in range(batch_size):
        for j in range(feature_dim):
            sequence = real_data[i, :, j].cpu().numpy()

            # Fit the sequence using Chebyshev polynomial
            coeffs = np.polynomial.chebyshev.chebfit(np.linspace(-1, 1, sequence_length), sequence, deg)

            # Differentiate the Chebyshev polynomial
            deriv_coeffs = np.polynomial.chebyshev.chebder(coeffs)

            # Evaluate the derivative polynomial
            deriv_sequence = np.polynomial.chebyshev.chebval(np.linspace(-1, 1, sequence_length), deriv_coeffs)

            # Store the result in the derivative tensor
            derivative[i, :, j] = torch.tensor(deriv_sequence)

    return derivative


'''
func:calcualte the diff of the data
input:real_data:torch.tensor and type
output:diff_grads:torch.tensor
'''
def calculate_diff_grads(real_data:torch.tensor,
                         data_t:torch.tensor,
                         type="center_diff",plt_show=False):
    '''

    :param real_data: [batch,100,2]
    :param data_t:     #data_t [batch,100,1]
    :param type: center_diff or chebyshev_appro
    :return:  grads:[batch,100,2],and figure
    '''
    delta_t=data_t[0,1,0]-data_t[0,0,0]

    if type=="center_diff":
        #diff_grads
        diff_grads=five_point_stencil(real_data,delta_t)
        if plt_show:
            plt.grid(True)
            plt.plot(data_t[0,:,0].cpu().detach().numpy(),real_data[0,:,0:1].cpu().detach().numpy(),
                     label="real_"+r"$\mathbf{z_{1}} $",color="red")
            plt.plot(data_t[0,:,0].cpu().detach().numpy(),real_data[0,:,1:2].cpu().detach().numpy(),
                     label="real_"+r"$\mathbf{z_{2}}$",color="blue")

            plt.plot(data_t[0,:,0].cpu().detach().numpy(),diff_grads[0,:,0:1].cpu().detach().numpy(),
                     label="real_grad_"+r"$\mathbf {\dot{z_{1}}} $",color="red",linestyle="--")
            plt.plot(data_t[0,:,0].cpu().detach().numpy(),diff_grads[0,:,1:2].cpu().detach().numpy(),
                        label="real_grad_"+r"$\mathbf {\dot{z_{2}}} $",color="blue",linestyle="--")

            plt.title("Center_diff For Gradients")
            plt.legend(loc='upper right')
            plt.xlabel("time (s)")
            plt.ylabel("value")


    elif type=="chebyshev_appro":
            diff_grads=chebyshev_appro(real_data,deg=10)
            if plt_show:
                plt.grid(True)
                plt.plot(data_t[0, :, 0].cpu().detach().numpy(), real_data[0, :, 0:1].cpu().detach().numpy(),
                        label="real_" + r"$\mathbf z_{1} $", color="blue")
                plt.plot(data_t[0, :, 0].cpu().detach().numpy(), real_data[0, :, 1:2].cpu().detach().numpy(),
                        label="real_" + r"$\mathbf z_{2}  $", color="red")

                plt.plot(data_t[0, :, 0].cpu().detach().numpy(), diff_grads[0, :, 0:1].cpu().detach().numpy(),
                        label="real_grad_" + r"$\mathbf {\dot{z_{1}}} $", color="red", linestyle="--")
                plt.plot(data_t[0, :, 0].cpu().detach().numpy(), diff_grads[0, :, 1:2].cpu().detach().numpy(),
                        label="real_grad_" + r"$\mathbf {\dot{z_{2}}} $", color="blue", linestyle="--")
                plt.title("Chebyshev For Gradients")
                plt.legend(loc='upper right')
                plt.xlabel("time (s)")
                plt.ylabel("value")



    return diff_grads



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
def compute_w_div_fouier_space(real,fake):
    '''
    computer the fouier space l2 distance
    :param real: [batch,100,2]
    :param fake: [batch,100,2]
    :return: [batch,scalar]
    '''
    #reduce the average
    reduce_average_1=torch.mean(real,dim=1)


    reduce_average_2=torch.mean(fake,dim=1)

    fft1 = gan_nerual.compute_spectrum(real-reduce_average_1.unsqueeze(1))
    fft2 = gan_nerual.compute_spectrum(fake-reduce_average_2.unsqueeze(1))

    # Compute the difference between the Fourier coefficients
    diff = fft1 - fft2
    # Compute the L2 norm of the difference
    distance= torch.nn.MSELoss()(fft1,fft2)



    return distance


def theil_u_statistic(y_true, y_pred):
    """
    Calculate Theil's U-statistic for predictions.

    Parameters:
    - y_true: Tensor of true values. Shape: (batch_size, sequence_length, 1)
    - y_pred: Tensor of predicted values. Same shape as y_true.

    Returns:
    - Theil's U-statistic.[0-1] 0 is perfect
    """
    # Calculate numerator (RMSE)
    numerator = torch.sqrt(torch.mean((y_true - y_pred) ** 2))

    # Calculate the denominators
    denominator_1 = torch.sqrt(torch.mean(y_true ** 2))
    denominator_2 = torch.sqrt(torch.mean(y_pred ** 2))

    # Calculate U-statistic
    u_statistic = numerator / (denominator_1 + denominator_2)

    return u_statistic

import matplotlib.colors as mcolors
matrix_step=0
def cost_matrix(real,fake,sampling_hz=49.5):
    '''
    :param real: [batch,100,2]
    :param fake: [batch,100,2]
    :return: [batch,50,50,2]
    '''
    global matrix_step
    matrix_step=matrix_step+1
    batch_size,_,vari=real.size()
    M = np.zeros((batch_size,50,50,2))
    real_fft_result = torch.fft.rfft(real, dim=1)
    fake_fft_result = torch.fft.rfft(fake, dim=1)

    P_real_freqs = np.fft.rfftfreq(100, d=1 / sampling_hz)[:]
    P_fake_freqs = np.fft.rfftfreq(100, d=1 / sampling_hz)[:]

    index=np.where(P_real_freqs>0)

    real_spectrum = real_fft_result[:,index[0],:].cpu().detach().numpy()
    real_frequencies = P_real_freqs[index[0]]
    fake_spectrum = fake_fft_result[:,index[0],:].cpu().detach().numpy()#[batch,50,2]
    fake_frequencies = P_fake_freqs[index[0]]


    #cost matrix
    for i in range(batch_size):
        for j in range(vari):
            value= ot.utils.dist(np.abs(real_spectrum[i,:,j]).reshape(-1,1), np.abs(fake_spectrum[i,:,j]).reshape(-1,1))
            M[i,:,:,j]=value
    print(matrix_step)

    # #plot
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    label_size = 20
    title_size = 20
    plt.hist(real_frequencies, bins=50, weights=np.abs(real_spectrum[0, :, 0]), alpha=0.6, label='Original Spectrum',
             color="g")
    plt.hist(fake_frequencies, bins=50, weights=np.abs(fake_spectrum[0, :, 1]), alpha=0.6, label='Target Spectrum',
             color="b")
    plt.xlabel("Frequency (Hz)", fontsize=label_size)
    plt.ylabel("Magnitude", fontsize=label_size)
    plt.legend()
    plt.grid("True")
    plt.title("Frequency Components Histogram", fontsize=title_size)
    plt.subplot(1, 2, 2)
    plt.imshow(M[0,:,:,0], cmap='viridis',norm=mcolors.LogNorm())
    plt.grid("True")
    plt.colorbar()
    plt.title("Cost Matrix Heatmap", fontsize=title_size)
    plt.xlabel("Original Frequencies", fontsize=label_size)
    plt.ylabel("Target Frequencies", fontsize=label_size)
    plt.tight_layout()
    plt.savefig("cost_matrix"+str(matrix_step)+".png")
    plt.close()

    return M
def infinity_norm(matrix):
    '''
    :param matrix: [50,50,2]
    :return: scalar
    '''
    #[2]
    inf_norm = [np.linalg.norm(matrix[:,:,i], ord=np.inf, axis=1) for i in range(2)]
    result = [np.max(inf_norm[i]) for i in range(2)] #two elements like[1.2,3.4]
    average_inf_norm = np.mean(result, axis=0)
    return average_inf_norm




if __name__ == "__main__":

    print("utils_func.py is loaded")


