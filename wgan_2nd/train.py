#this file for w_gan
import utlis_2nd.utlis_funcs as uf
import utlis_2nd.gan_nerual as gan_nerual
import torch
import torch.optim as optim
import os
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.nn as nn
from tqdm import tqdm
class CustomDataset(Dataset):
    def __init__(self, file_path):
        # readt .pt
        data = torch.load(file_path)
        self.data = data['data']
        self.label = data['label_csv']
        self.length = len(self.data)

    def __getitem__(self, index):
        # get data&label
        #label is a string of csv number
        data = self.data[index]
        label = self.label[index]
        # pre-processing
        return data, label

    def __len__(self):
        return self.length

#set_default_dtype float64
torch.set_default_dtype(torch.float64)
#get loader data
path="easy_center_dataset"
train_loader = torch.load(path+'/center_train.pth')
test_loader = torch.load(path+'/center_test.pth')
#set seed
uf.set_seed(42)
#config
config={
        'batch_size':2,
        'g_neural_network_width':100,
        'zdimension_Gap':10,
        "save_plot_tb_path":"../tb_info/wgan_2nd_trainA_100_5",
        "lamba_ini":10,
        "lamba_grad":1,
        "lamba_gp":10,
        "prior_knowledge":{"basis_1": "x**0", "basis_2": "sin", "basis_3": "cos"},
        "freq_threshold":1e-3,
        }
#deivice
device='cuda'
#writer
writer = SummaryWriter(config['save_plot_tb_path'])
def train(D,G,G_omega,z_dimen,num_epoch=1000):
    global fake, data_t, real, label
    d_loss_list = []
    div_gp_list = []
    energy_list = []
    wasses_d_list=[]
    g_loss_list=[]
    ini_loss_list=[]
    deriva_loss_list=[]
    #dict_score
    dict_score={"real_trap":[],"fake_trap":[],
                "real_out":[],"fake_out":[]}
    #optimizer
    d_optimizer=optim.Adam(D.parameters(),lr=1e-5)
    g_optimizer=optim.Adam(G.parameters(),lr=1e-5)
    g_omega_optimizer=optim.Adam(G_omega.parameters(),lr=1e-3)
    #criterion
    criterion_ini = nn.MSELoss()
    criterion_deriva = nn.MSELoss()
    criterion_freq=nn.MSELoss()
    #falg--state switch
    Flag_train_genertor=0
    #train
    for epoch in range(num_epoch):
        #record time
        start_time=time.time()
        for i,(batch_data,label) in enumerate(train_loader):

           #batch_size[2,100,9]
           num_batch=batch_data.size(0)

           #every iteration：we pick the basis_function and record them
           left_matrix,symbol_matrix=gan_nerual.Get_basis_function_info(config['prior_knowledge'])

           #get the condition_data and data_t and dict_str_solu
           #real_condition[batch,101,2]
           real_condition,data_t,dict_str_solu=gan_nerual.convert_data(batch_data,label)

           #real_data
           real=batch_data[:,:,7:9].to("cuda")
           real.requires_grad=True

           #omega_value
           omega_value=G_omega(real_condition)

           # real out score scalar about trans_condition
           score_real_out=D(real,condition_info=real_condition)

           #z-noise sample for fake data
           z=torch.randn(num_batch,z_dimen).to(device)

           #generate by inference ---linear combination [batch,6]
           fake_coeffs=G(z,real_condition,omega_value)
           updated_symbol_list,fake_data,fake_condition,left_matrix=gan_nerual.multiply_matrix(fake_coeffs,
                                                                               omega_value,
                                                                               left_matrix,
                                                                               symbol_matrix)

           #fake-out score
           #critic for data and (grads and ini_condition)
           score_fake_out=D(fake_data,condition_info=fake_condition)

           div_gp=gan_nerual.compute_w_div(real,
                                           score_real_out,
                                           fake_data,
                                           score_fake_out)
           #wasserstein_distance
           wasses_distance=torch.mean(score_real_out)-torch.mean(score_fake_out)

           #loss list
           d_loss=-(torch.mean(score_real_out)-torch.mean(score_fake_out))+div_gp
           d_loss_list.append(d_loss)
           div_gp_list.append(div_gp)
           wasses_d_list.append(wasses_distance)

           # #save fig and visualize fake
           uf.plot_critic_tensor_change(config['save_plot_tb_path'],writer,
                                        data_t,
                                        fake_data,real,
                                        updated_symbol_list,
                                        left_matrix,
                                        fake_coeffs,
                                        dict_str_solu,
                                        )

           #optimizer
           d_optimizer.zero_grad()
           d_loss.backward()
           d_optimizer.step()

           for j in range(100):

               if Flag_train_genertor == 0:
                   # pretained the omega_neural
                   # g_omega_optimizer
                   real_condition_4omega = real_condition.clone().detach()
                   real_condition_4omega.require_grad=True
                   #cal the omega---fake--omegae
                   omega_value = G_omega(real_condition_4omega)

                   #compare the fourier domain's difference
                   real_freq = gan_nerual.compute_spectrum(real)
                   generator_freq = omega_value

                   # g_omega_loss
                   g_omega_freq_loss = criterion_freq(generator_freq, real_freq)

                   g_omega_optimizer.zero_grad()
                   # optimizer
                   g_omega_freq_loss.backward()

                   # omega _loss is depending on freq! so we need to calculate the freq
                   g_omega_optimizer.step()

                   if (g_omega_freq_loss.item ()< config["freq_threshold"]):
                       Flag_train_genertor = 1
                       writer.add_scalar('Model_Switch', Flag_train_genertor, epoch)  # switch state

               else:

                   Flag_train_genertor=0
                   writer.add_scalar('Model_Switch', Flag_train_genertor, epoch)  # switch state
                   # train the generator
                   g_optimizer.zero_grad()
                   real_condition_4omega = real_condition.detach()
                   real_initial = real_condition_4omega[:, 0, 0:2] # shape:[batch,1,2]

                   # z -noise
                   z = torch.randn(num_batch, z_dimen).to("cuda")

                   #free the G_omega :freeze
                   omega_value = G_omega(real_condition_4omega)

                   #fake data
                   fake_coeffs = G(z, real_condition_4omega, omega_value)
                   updated_symbol_list, fake_data, fake_condition, basis_matrix = gan_nerual.multiply_matrix(
                                                                                   fake_coeffs,
                                                                                   omega_value,
                                                                                   left_matrix,
                                                                                   symbol_matrix)
                   # real_grads
                   real_grads= uf.calculate_diff_grads(real, data_t)

                   # fake_inital
                   fake_inital = fake_condition[:, 0, 0:2]  # [batch,2]

                   # fake_grad_condition
                   fake_grads = fake_condition[:, 1:, 0:2]  # [batch,100,2]

                   # score_fake_out:need to freeze the D
                   fake_data.detach_()
                   fake_condition.detach_()
                   score_fake_out = D(fake_data, fake_condition)

                   # ini_condition loss
                   ini_loss = criterion_ini(fake_inital, real_initial)
                   ini_loss_list.append(ini_loss)

                   # derivative loss
                   deriva_loss = criterion_deriva(fake_grads, real_grads)
                   deriva_loss_list.append(deriva_loss)

                   #calculate the loss
                   g_loss = -torch.mean(score_fake_out)+config["lamba_ini"]*ini_loss+\
                            config["lamba_grad"]*deriva_loss
                   g_loss_list.append(g_loss)
                   # generator optimizer

                   g_loss.backward()
                   g_optimizer.step()
                   Flag_train_genertor=0



        #save for the avg_epoch_loss
        epoch_d_loss=sum(d_loss_list) / len(d_loss_list)
        epoch_g_loss = sum(g_loss_list) / (len(g_loss_list))
        print(f"***dloss&gloss***{epoch}",epoch_d_loss,epoch_g_loss)
        writer.add_scalars("all_loss",{"d_loss":epoch_d_loss,
                                       "g_loss":epoch_g_loss,
                                       "all_loss":epoch_d_loss+epoch_g_loss},epoch)
        writer.add_scalars("train_g_loss",{"ini_loss":sum(ini_loss_list) / (len(ini_loss_list)),
                                    "deriva_loss":sum(deriva_loss_list) / (len(ini_loss_list))},epoch)

        epoch_div=sum(div_gp_list) / len(div_gp_list)
        writer.add_scalar("div_gp", epoch_div, epoch)

        wasses_distance=sum(wasses_d_list) / len(wasses_d_list)
        writer.add_scalar("wasses_distance", wasses_distance, epoch)

        #save the scalars

        # writer.add_scalars("critic_score",{"real_out":sum(dict_score["real_out"]) / len(dict_score["real_out"]),
        #                                    "fake_out":sum(dict_score["fake_out"]) / len(dict_score["fake_out"])
        #                             },epoch)
        #
        # #save the fake and real data
        #
        # #embedding batch_data
        # for j in range(config["batch_size"]):
        #
        #     save_fake_tensors=torch.concatenate((data_t[j,:,0:1],fake[j,:,0:2]),dim=1)
        #     writer.add_embedding(save_fake_tensors,
        #                          metadata = [[f"data_t{i}",f"fake_z1_t{i} ", f"fake_z2_t {i}",label[j]] for i in range(100)],
        #                          tag = 'fake_data'+str(j)+"label="+str(label[j]), global_step = epoch)
        #
        #     save_real_tensors=torch.concatenate((data_t[j,:,0:1],real[j,:,0:2]),dim=1)
        #     writer.add_embedding(save_real_tensors,
        #                          metadata=[[f"data_t{i}", f"real_z1_t {i}", f"real_z2_t {i}",label[j]] for i in range(100)],
        #                          tag='real_data'+str(j)+"label="+str(label[j]), global_step=epoch)

        #check the epoch time
        final_time = time.time()
        print("epoch_time",final_time-start_time,flush=True)

        # #every 100epoch save the checkpoint
        if epoch % 100==0:
            checkpoint={
                "epoch":epoch,
                "D_model_state_dict":D.state_dict(),
                "G_model_state_dict":G.state_dict(),
                "optimizer_state_dict":d_optimizer.state_dict(),
                "loss":0
            }
        torch.save(checkpoint,config["save_plot_tb_path"]+'/checkpoint.pth')

        # close
    writer.close()

'''
func: evaluate the model
param: model_path
return: fid score and mse
'''

def eval_model(model_path:str,save_path:str):
    uf.set_seed(42)
    writer=SummaryWriter(save_path)
    #load the model
    checkpoint=torch.load(model_path)
    D=gan_nerual.Discriminator().to(device)
    G=gan_nerual.Generator(config).to(device)
    D.load_state_dict(checkpoint["D_model_state_dict"])
    G.load_state_dict(checkpoint["G_model_state_dict"])
    G.eval()
    D.eval()
    model = nn.DataParallel(G)
    #calculate fid score and mse
    fid=[]
    mse=[]

    for i, (batch_data, label) in enumerate(test_loader):

        #label to str
        label=str(label)

        # input the z-noise
        z = torch.randn(config["batch_size"],config['zdimension_Gap']).to(device)
        data_t = batch_data[:, :, 6].clone()
        data_t = data_t.unsqueeze(dim=2)  # shape:[batch,100,1]
        data_t = data_t.to(device).requires_grad_(True)
        real = batch_data[:, :, 7:9].to(device)


        # generate the fake data
        fake, energy, coeffs, basis_matrix, basis_str = G(z, data_t)


        for j in range(config["batch_size"]):
            #save the fake and real data
            save_fake_tensors = torch.concatenate((data_t[j, :, 0:1], fake[j, :, 0:2]),
                                                  dim=1)

            save_real_tensors = torch.concatenate((data_t[j, :, 0:1], real[j, :, 0:2]),
                                                  dim=1)
            writer.add_embedding(save_real_tensors,
                                 metadata=[[f"data_t{i}", f"real_z1_t {i}", f"real_z2_t {i}",label[j]] for i in range(100)],
                                 tag='real_data' + str(j)+"label="+label[j], global_step=j)

            writer.add_embedding(save_fake_tensors,
                                 metadata=[[f"data_t{i}", f"fake_z1_t{i} ", f"fake_z2_t {i}",label[j]] for i in range(100)],
                                 tag='fake_data' + str(j)+"label="+label[j], global_step=j)
            #z1_t and z2_t
            fid.append(uf.calculate_fid_score(real[j,:,0:2],fake[j,:,0:2]))
            mse.append(uf.calculate_mse(real[j,:,0:2], fake[j,:,0:2]))
        #save the fake and real data

    fid_score=sum(fid)/len(fid)
    mse=sum(mse)/len(mse)
    print("**fid_score",fid_score)
    print("**mse",mse)
    writer.add_scalar("eval_fid_score",fid_score)
    writer.add_scalar("eval_mse",mse)

    return fid_score,mse



if __name__=='__main__':

    #dp the train
    print("start train",flush=True)
    print("the prior knowledge is",config["prior_knowledge"],flush=True)
    D=gan_nerual.Discriminator().to(device)
    G=gan_nerual.Generator(config,basis_num=3,omega_value=0).to(device)
    G_omega=gan_nerual.omega_generator(input_dim=202,output_dim=2).to(device)
    D = torch.nn.DataParallel(D, device_ids=[0])
    G = torch.nn.DataParallel(G, device_ids=[0])
    G_omega = torch.nn.DataParallel(G_omega, device_ids=[0])
    train(D,G,G_omega,z_dimen=config['zdimension_Gap'],num_epoch=1000)
    #eval_model(model_path="../tb_info/wgan_2nd/checkpoint.pth",
               #save_path="../tb_info/wgan_2nd_eval")



