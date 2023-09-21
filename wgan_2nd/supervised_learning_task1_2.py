#co-train
# this is a task for supervised learning
# supervised the fourier transform
# this file for w_gan
import utlis_2nd.utlis_funcs as uf
import utlis_2nd.gan_nerual as gan_nerual
import utlis_2nd.co_train as co_train
import torch
import torch.optim as optim
import os
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.nn as nn
import ot
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
        # label is a string of csv number
        data = self.data[index]
        label = self.label[index]
        # pre-processing
        return data, label

    def __len__(self):
        return self.length

# set_default_dtype float64
torch.set_default_dtype(torch.float64)
# get loader data
path = "complex_center_dataset"
train_loader = torch.load(path + '/complex_center_train.pth')
valid_loader = torch.load(path + '/complex_center_valid.pth')
test_loader = torch.load(path + '/complex_center_test.pth')

# config
config = {
    "batch_size": 32,
    "g_neural_network_width": 512,
    "save_plot_tb_path": "../tb_info/wgan_2nd_trainA_100_test_env",
    "zdimension_Gap": 0,
    "lamba_ini": 1,
    "lamba_grad": 1,
    "lamba_gp": 1,
    "lamba_fourier": 1,
    "prior_knowledge": {"basis_1": "x**0", "basis_2": "sin", "basis_3": "cos"},

    "S_I_lr": 1e-5,
    "S_Omega_lr": 1e-4,
    "grads_types": {"boundary": 3, "inner": 5},  # boundary:forward and backward;inner:five points
    "beta": 2,
    "freq_numbers": 3,
    "training_data_path": "/training_data",
    "training_omega_data_path": "/training_omega_data",
    "training_inference_path": "/training_inference_data",
    "eval_data_path": "/eval_data",
    "eval_omega_data_path": "/eval_omega_data",
    "eval_inference_path": "/eval_inference_data",
    "seed":42,
    "Omega_num_epoch":200,
    "Inference_num_epoch":10,
    "co_train_time" : 100,
    "infer_minimum":1e-1,

}

device = 'cuda'
# writer
writer = SummaryWriter(config['save_plot_tb_path'])
time_dynamic=0
omega_time=0
def train():
    # set seed
    uf.set_seed(config["seed"])
    global time_dynamic
    global omega_time
    print("co-train_omega")
    for co_train_epoch in range(config["co_train_time"]):

        eval_infer_value=co_train_actor.train_inference_neural()
        time_dynamic=time_dynamic+config["Inference_num_epoch"]
        print(time_dynamic)

        if(eval_infer_value>config["infer_minimum"]):
            flag=0
            omega_time=omega_time+1
            co_train_actor.train_omega_neural()#蒸馏温度逐渐升高
            time_dynamic = time_dynamic+(omega_time * config["Omega_num_epoch"])
            writer.add_scalar("co-train_epoch", flag, time_dynamic)
            print(time_dynamic)

        if(eval_infer_value<=config["infer_minimum"]):
            flag=1
            writer.add_scalar("co-train_epoch",flag, time_dynamic)
            print("statisfy")
            break
    print("done")

def test_model():
    pass


if __name__ == '__main__':
    uf.set_seed(config["seed"])

    print("start train_Supervised_learning", flush=True)
    num_gpus = torch.cuda.device_count()
    print(f"Total available GPUs: {num_gpus}")
    print("the prior knowledge is", config["prior_knowledge"], flush=True)
    #co_train
    S_I = gan_nerual.Generator(input_dim=400,
                               output_dim=(2 * config["freq_numbers"] + 1) * 2).to(device)
    S_Omega = gan_nerual.omega_generator(input_dim=400,
                                         output_dim=2 * config["freq_numbers"]).to(device)
    co_train_actor = co_train.train_init(S_I,S_Omega,config, train_loader,
                                         valid_loader, test_loader, writer)
    # dp the train
    co_train_actor.S_Omega = torch.nn.DataParallel(co_train_actor.S_Omega, device_ids=[0])
    co_train_actor.S_I = torch.nn.DataParallel(co_train_actor.S_I, device_ids=[0])
    # record
    writer.add_text(str(config), config["save_plot_tb_path"])
    if not os.path.exists(config["save_plot_tb_path"] + config["training_data_path"]):
        os.makedirs(config["save_plot_tb_path"] + config["training_data_path"])
        os.makedirs(config["save_plot_tb_path"] + config["training_data_path"] + config["training_omega_data_path"])
        os.makedirs(config["save_plot_tb_path"] + config["training_data_path"] + config["training_inference_path"])
        print("make the train dir")
    if not os.path.exists(config["save_plot_tb_path"] + config["eval_data_path"]):
        os.makedirs(config["save_plot_tb_path"] + config["eval_data_path"])
        os.makedirs(config["save_plot_tb_path"] + config["eval_data_path"] + config["eval_omega_data_path"])
        os.makedirs(config["save_plot_tb_path"] + config["eval_data_path"] + config["eval_inference_path"])
        print("make the eval dir")

    train()






