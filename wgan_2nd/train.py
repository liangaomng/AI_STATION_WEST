#this file for f_gan
import sys

import utlis_2nd.utlis_funcs as uf
import utlis_2nd.gan_nerual as gan_nerual
import torch
import torch.optim as optim
import os
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
class CustomDataset(Dataset):
    def __init__(self, file_path):
        # 读取.pt文件
        data = torch.load(file_path)
        self.data = data['data']
        self.label = data['label_csv']
        self.length = len(self.data)

    def __getitem__(self, index):
        # 获取数据和标签
        data = self.data[index]
        label = self.label[index]

        # 数据预处理操作

        return data, label

    def __len__(self):
        return self.length

torch.set_default_dtype(torch.float64)
train_loader = torch.load("2nd_dataset/2ndtrain.pth")
test_loader = torch.load("2nd_dataset/2ndtest.pth")
uf.set_seed(42)
config={'batch_size':64,
        'g_neural_network_width':100,
        'zdimension_Gap':100,
        "save_plot_tb_path":"../tbinfo/wgan_2nd"}
device='cuda'


writer = SummaryWriter("../tb_info/wgan_2nd")

def train(D,G,z_dimen,num_epoch=100):
    d_loss_list = []
    div_gp_list = []
    energy_list = []
    coeffs_list= []
    d_optimizer=optim.Adam(D.parameters(),lr=1e-4)
    g_optimizer=optim.Adam(G.parameters(),lr=1e-4)

    for epoch in range(num_epoch):
        for i,(batch_data,label) in enumerate(train_loader):

           data_t=batch_data[:,:,6].clone()
           data_t=data_t.unsqueeze(dim=2) #shape:[batch,100,1]
           #critic for real data are z1_t z2_t
           #real out score
           real=batch_data[:,:,7:9].to(device).requires_grad_(True) #[batach,100,2]
           real_out=D(real)
           #critic for fake data
           #z-noise
           z=torch.randn(config["batch_size"],z_dimen).to(device)
           fake,energy,coeffs,basis_matrix,basis_str=G(z,data_t)
           energy=torch.mean(energy)
           #fake-out score
           fake_out=D(fake)
           div_gp=gan_nerual.compute_w_div(real,real_out,fake,fake_out)

           #loss
           d_loss=-(torch.mean(real_out)-torch.mean(fake_out))+div_gp
           d_loss_list.append(d_loss)
           div_gp_list.append(div_gp)
           energy_list.append(energy)

           #save fig and visualize
           uf.plot_critic_tensor_change(config['save_plot_tb_path'],writer,
                                        d_loss,data_t,
                                        fake,real,
                                        basis_str,basis_matrix,coeffs,
                                        energy)
           exit()

           #optimizer
           d_optimizer.zero_grad()
           d_loss.backward(retain_graph=True)
           d_optimizer.step()

        #save for the epoch_loss
        epoch_loss=sum(d_loss_list) / len(d_loss_list)
        writer.add_scalar("critic_loss",epoch_loss, epoch)

        epoch_div=sum(div_gp_list) / len(div_gp_list)
        writer.add_scalar("div_gp", epoch_div, epoch)

        epoch_energy=sum(energy_list) / len(energy_list)
        writer.add_scalar("energy", epoch_energy, epoch)





if __name__=='__main__':

    D=gan_nerual.Discriminator().to(device)
    G=gan_nerual.Generator(config).to(device)
    train(D,G,z_dimen=100)



