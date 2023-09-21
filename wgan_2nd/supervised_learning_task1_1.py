# this is a task for supervised learning
# supervised the fourier transform
# this file for w_gan
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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

torch.set_default_dtype(torch.float64)
# get loader data
path = "complex_center_dataset"
train_loader = torch.load(path + '/complex_center_train.pth')
valid_loader = torch.load(path + '/complex_center_valid.pth')
test_loader = torch.load(path + '/complex_center_test.pth')
# set seed
uf.set_seed(42)

# config
config = {
    "batch_size": 32,
    "g_neural_network_width": 512,
    "save_plot_tb_path": "../tb_info/wgan_2nd_trainA_100_3_env",
    "zdimension_Gap": 0,
    "lamba_ini": 1,
    "lamba_grad": 1,
    "lamba_gp": 1,
    "lamba_fourier": 1,
    "prior_knowledge": {"basis_1": "x**0", "basis_2": "sin", "basis_3": "cos"},
    "S_I_epoch": 10000,
    "S_Omega_epoch": 5000,
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

}

# plot for visual the loss
S_Omega_numpy = np.zeros((4, config["S_Omega_epoch"]))
S_I_numpy = np.zeros((4, config["S_I_epoch"]))
# deivice
device = 'cuda'
# writer
writer = SummaryWriter(config['save_plot_tb_path'])


def train(S_I, S_Omega, Omgea_num_epoch=1000, I_num_epoch=1000):
    # optimizer
    S_I_optimizer = optim.Adam(S_I.parameters(), lr=config["S_I_lr"])
    S_Omega_optimizer = optim.Adam(S_Omega.parameters(), lr=config["S_Omega_lr"])
    # criterion
    criterion_inference = nn.MSELoss()
    criterion_ini = nn.MSELoss()
    criterion_grad = nn.MSELoss()
    criterion_fourier = nn.MSELoss()
    # loss list
    g_omega_freq_loss_list = []
    inference_loss_list = []
    ini_loss_list = []
    grad_loss_list = []
    fourier_loss_list = []
    mse_loss_list = []

    # supervised learning-omega
    S_I_step = 0
    S_Omega_step = 0
    # prepare:data_t
    numpy_t = np.linspace(0, 2, 100)
    numpy_t = numpy_t.reshape(100, 1)
    data_t = torch.from_numpy(numpy_t).float().to(device)
    data_t = data_t.unsqueeze(0).expand(32, -1, 1)

    if os.path.exists(config["save_plot_tb_path"] + '/omega_checkpoint.pth'):
        checkpoint = torch.load(config["save_plot_tb_path"] + '/omega_checkpoint.pth')
        S_Omega.load_state_dict(checkpoint['S_Omega_model_state_dict'])
        S_Omega_optimizer.load_state_dict(checkpoint['S_Omega_optimizer_state_dict'])
        print("load Supervised model successfully")
        print("trainning_inference")

        for epoch in range(I_num_epoch):
            start_time = time.time()
            for i, (batch_data, label) in enumerate(train_loader):
                S_I_step += 1
                # get the condition_data and data_t and dict_str_solu
                # real_data
                real = batch_data[:, :, 7:9].to(device)
                # real_condition[batch,200,2]
                real_condition, _ = gan_nerual.convert_data(real, data_t, label, step=S_I_step)

                # omega_value
                with torch.no_grad():
                    generator_freq = S_Omega(real_condition)

                # generator_freq[batch,numbers,2]
                generator_freq = generator_freq.reshape(-1, config["freq_numbers"], 2)

                # compare the fourier domain's difference
                real_freq = gan_nerual.compute_spectrum(real,
                                                        beta=config["beta"],
                                                        freq_number=config["freq_numbers"],
                                                        train_step=S_I_step,
                                                        path=config["save_plot_tb_path"] + config[
                                                            "training_data_path"] + config[
                                                                 "training_inference_path"] + "/real")

                # inference loss
                fake_coeffs = S_I(real_condition)
                fake_coeffs = fake_coeffs.reshape(config["batch_size"], -1, 2)

                if S_I_step == 1:
                    left_matrix, symbol_matrix = gan_nerual.Get_basis_function_info(config['prior_knowledge'])

                updated_symbol_list_z1, left_matrix, fake_data, fake_condition = gan_nerual.return_torch_version_matrix(
                    fake_coeffs,
                    generator_freq,
                    symbol_matrix)
                fake_freq = gan_nerual.compute_spectrum(fake_data,
                                                        beta=config["beta"],
                                                        freq_number=config["freq_numbers"],
                                                        train_step=S_I_step,
                                                        path=config["save_plot_tb_path"] + config[
                                                            "training_data_path"] + config[
                                                                 "training_inference_path"] + "/fake")

                # save for the data---PINN
                mse_loss = criterion_inference(fake_data, real)
                mse_loss_list.append(mse_loss)
                fouier_loss = criterion_fourier(fake_freq, real_freq)
                fourier_loss_list.append(fouier_loss)
                # just for record
                gradient_loss = criterion_grad(fake_condition[:, 100:, :], real_condition[:, 100:, :])
                grad_loss_list.append(gradient_loss)
                ini_loss = criterion_ini(fake_condition[:, 0, :], real_condition[:, 0, :])
                ini_loss_list.append(ini_loss)

                infer_loss = mse_loss + config["lamba_fourier"] * fouier_loss
                inference_loss_list.append(infer_loss)
                # optimizer
                S_I_optimizer.zero_grad()
                # loss
                infer_loss.backward()
                S_I_optimizer.step()

            final_time = time.time()

            epoch_inference_loss = sum(inference_loss_list) / len(inference_loss_list)
            epoch_fourier_loss = sum(fourier_loss_list) / len(fourier_loss_list)
            epoch_ini_loss = sum(ini_loss_list) / len(ini_loss_list)
            epoch_grad_loss = sum(grad_loss_list) / len(grad_loss_list)
            epoch_mse_loss = sum(mse_loss_list) / len(mse_loss_list)
            writer.add_scalars("trian_inference_loss", {"inference_loss": epoch_inference_loss,
                                                        "fourier_loss": epoch_fourier_loss,
                                                        "mse_loss": epoch_mse_loss}, epoch)
            writer.add_scalars("trian_inference_record_not_loss", {"ini_loss": epoch_ini_loss,
                                                                   "grad_loss": epoch_grad_loss}, epoch)

            inference_loss_list = []
            fourier_loss_list = []
            ini_loss_list = []
            grad_loss_list = []
            mse_loss_list = []


            print(f"loss{epoch_inference_loss.item()}" + f"_epoch{epoch}" + "epoch_time" + \
                  f"{final_time - start_time}", flush=True)
            # #every 100epoch save the checkpoint
            if epoch % 100 == 0:
                checkpoint = {
                    "epoch": epoch,
                    "S_I_model_state_dict": S_I.state_dict(),
                    'S_Omega_optimizer_state_dict': S_I_optimizer.state_dict(),
                    "loss": epoch_inference_loss,
                }
                torch.save(checkpoint, config["save_plot_tb_path"] + '/Inference_checkpoint.pth')
            with torch.no_grad():
                eval_inference_model(eval_data=valid_loader, model_I=S_I, model_O=S_Omega, data_t=data_t,
                                     eval_epoch=epoch)

    else:
        print("trainning_omega")
        for epoch in range(Omgea_num_epoch):
            start_time = time.time()
            for i, (batch_data, label) in enumerate(train_loader):
                S_Omega_step += 1

                # get the condition_data and data_t and dict_str_solu

                # real_data
                real = batch_data[:, :, 7:9].to("cuda")
                label = label.to("cuda")

                # real_condition[batch,200,2]
                real_condition, _ = gan_nerual.convert_data(real, data_t, label, step=S_Omega_step)

                # omega_value
                generator_freq = S_Omega(real_condition)

                # generator_freq[batch,numbers,2]
                generator_freq = generator_freq.reshape(-1, config["freq_numbers"], 2)

                # compare the fourier domain's difference
                real_freq = gan_nerual.compute_spectrum(real,
                                                        beta=config["beta"],
                                                        freq_number=config["freq_numbers"],
                                                        train_step=S_Omega_step,
                                                        path=config["save_plot_tb_path"] + config[
                                                            "training_data_path"] + config[
                                                                 "training_omega_data_path"] + "\real")

                # g_omega_loss
                g_omega_freq_loss = criterion_fourier(generator_freq, real_freq)
                g_omega_freq_loss_list.append(g_omega_freq_loss)

                # save for the data
                # optimizer
                S_Omega_optimizer.zero_grad()
                g_omega_freq_loss.backward()
                S_Omega_optimizer.step()

            final_time = time.time()
            epoch_g_omega_freq_loss = sum(g_omega_freq_loss_list) / len(g_omega_freq_loss_list)
            writer.add_scalar("train_omega_freq_loss", epoch_g_omega_freq_loss, epoch)
            g_omega_freq_loss_list = []
            print(f"loss{epoch_g_omega_freq_loss.item()}" + f"_epoch{epoch}" + "epoch_time" + \
                  f"{final_time - start_time}", flush=True)
            # #every 100epoch save the checkpoint
            if epoch % 100 == 0:
                checkpoint = {
                    "epoch": epoch,
                    "S_Omega_model_state_dict": S_Omega.state_dict(),
                    'S_Omega_optimizer_state_dict': S_Omega_optimizer.state_dict(),
                    "loss": epoch_g_omega_freq_loss,
                }
                torch.save(checkpoint, config["save_plot_tb_path"] + '/omega_checkpoint.pth')
            with torch.no_grad():
                eval_omega_model(eval_data=valid_loader, model=S_Omega, data_t=data_t, eval_epoch=epoch)


'''
func: evaluate the model
'''


def eval_inference_model(eval_data, model_I, model_O, data_t, eval_epoch):
    '''

    :param eval_data:
    :param model_I:
    :param model_O:
    :param data_t:
    :param eval_epoch:
    :return:
    '''
    S_I_eval_step = 0
    criterion_fourier = nn.MSELoss()
    eval_freq_loss_list = []
    eval_data_loss_list = []
    eval_grad_loss_list = []
    eval_u_stat_freq_list = []
    eval_u_stat_data_list = []
    eval_u_stat_grad_list = []
    eval_infinite_cost_list = []
    for i, (batch_data, label) in enumerate(eval_data):
        S_I_eval_step += 1
        # get the condition_data and data_t and dict_str_solu
        # real_data
        real = batch_data[:, :, 7:9].to(device)
        # real_condition[batch,200,2]
        real_condition, _ = gan_nerual.convert_data(real, data_t, label, step=S_I_eval_step)
        if S_I_eval_step == 1:
            left_matrix, symbol_matrix = gan_nerual.Get_basis_function_info(config['prior_knowledge'])
        # omega_value
        with torch.no_grad():
            generator_freq = model_O(real_condition)

        # generator_freq[batch,numbers,2]
        generator_freq = generator_freq.reshape(-1, config["freq_numbers"], 2)

        # inference loss
        fake_coeffs = model_I(real_condition)
        fake_coeffs = fake_coeffs.reshape(-1, (2 * config["freq_numbers"] + 1), 2)
        updated_symbol_list, left_matrix, fake_data, fake_condition = gan_nerual.return_torch_version_matrix(
            fake_coeffs,
            generator_freq,
            symbol_matrix
        )
        real_freq = gan_nerual.compute_spectrum(real,
                                                beta=config["beta"],
                                                freq_number=config["freq_numbers"],
                                                train_step=S_I_eval_step,
                                                path=config["save_plot_tb_path"] + config["eval_data_path"] + config[
                                                    "eval_inference_path"]
                                                     + "\real")
        pred_freq = gan_nerual.compute_spectrum(fake_data,
                                                beta=config["beta"],
                                                freq_number=config["freq_numbers"],
                                                train_step=S_I_eval_step,
                                                path=config["save_plot_tb_path"] + config["eval_data_path"] + config[
                                                    "eval_inference_path"]
                                                     + "\fake")

        # g_omega_loss
        g_omega_freq_loss = criterion_fourier(pred_freq, real_freq)
        eval_freq_loss_list.append(g_omega_freq_loss)
        # data_loss
        data_loss = criterion_fourier(fake_data, real)
        eval_data_loss_list.append(data_loss)
        # grad_loss
        grad_loss = criterion_fourier(fake_condition[:, 100:, :], real_condition[:, 100:, :])
        eval_grad_loss_list.append(grad_loss)
        # u_stat
        u_stat_freq = uf.theil_u_statistic(pred_freq, real_freq)
        u_stat_data = uf.theil_u_statistic(fake_data, real)
        u_stat_condition = uf.theil_u_statistic(fake_condition[:, 100:, :], real_condition[:, 100:, :])
        eval_u_stat_freq_list.append(u_stat_freq)
        eval_u_stat_data_list.append(u_stat_data)
        eval_u_stat_grad_list.append(u_stat_condition)
        # cost_matirx infinite_norm
        cost_matrix = uf.cost_matrix(real, fake_data)  # [batch,50,50,2]

        infinite_norm = [uf.infinity_norm(cost_matrix[i]) for i in range(cost_matrix.shape[0])]
        infinite_norm_avr = np.mean(infinite_norm)

        eval_infinite_cost_list.append(infinite_norm_avr)

    eval_freq_loss = sum(eval_freq_loss_list) / len(eval_freq_loss_list)
    eval_data_loss = sum(eval_data_loss_list) / len(eval_data_loss_list)
    eval_grad_loss = sum(eval_grad_loss_list) / len(eval_grad_loss_list)
    eval_u_stat_freq = sum(eval_u_stat_freq_list) / len(eval_u_stat_freq_list)
    eval_u_stat_data = sum(eval_u_stat_data_list) / len(eval_u_stat_data_list)
    eval_u_stat_grad = sum(eval_u_stat_grad_list) / len(eval_u_stat_grad_list)
    eval_infinite_cost = sum(eval_infinite_cost_list) / len(eval_infinite_cost_list)

    # save the eval result
    writer.add_scalars('eval_inference', {'eval_freq_loss': eval_freq_loss,
                                          'eval_data_loss': eval_data_loss,
                                          'eval_grad_loss': eval_grad_loss,
                                          'eval_u_stat_freq': eval_u_stat_freq,
                                          'eval_u_stat_data': eval_u_stat_data,
                                          'eval_u_stat_grad': eval_u_stat_grad,
                                          'eval_infinite_cost': eval_infinite_cost
                                          }, eval_epoch)


def eval_omega_model(eval_data, model, data_t, eval_epoch):
    '''
    :param eval_data: eval data
     return mse and u-stat
    '''
    eval_step = 0
    criterion_fourier = nn.MSELoss()
    eval_mse_loss_list = []
    eval_u_stat_list = []

    for i, (batch_data, label) in enumerate(eval_data):
        eval_step = eval_step + 1
        # get the condition_data and data_t and dict_str_solu

        # real_data
        real = batch_data[:, :, 7:9].to(device)

        # real_condition[batch,200,2]
        real_condition, _ = gan_nerual.convert_data(real, data_t, label, step=eval_step)

        # omega_value
        generator_freq = model(real_condition)

        # generator_freq[batch,numbers,2]
        generator_freq = generator_freq.reshape(-1, config["freq_numbers"], 2)
        # compare the fourier domain's difference
        real_freq = gan_nerual.compute_spectrum(real,
                                                beta=config["beta"],
                                                freq_number=config["freq_numbers"],
                                                train_step=eval_step,
                                                path=config["save_plot_tb_path"] + config["eval_data_path"] + config[
                                                    "eval_omega_data_path"] + '\real')

        # g_omega_loss
        g_omega_freq_loss = criterion_fourier(generator_freq, real_freq)
        eval_mse_loss_list.append(g_omega_freq_loss)
        # u_stat
        u_stat = uf.theil_u_statistic(generator_freq, real_freq)
        eval_u_stat_list.append(u_stat)

    eval_mse_loss = sum(eval_mse_loss_list) / len(eval_mse_loss_list)
    eval_u = sum(eval_u_stat_list) / len(eval_u_stat_list)

    writer.add_scalars("eval_omega", {"mse_loss": eval_mse_loss,
                                      "u_stat": eval_u,
                                      }, eval_epoch)


def test_model():
    pass


if __name__ == '__main__':

    print("start train_Supervised_learning", flush=True)
    num_gpus = torch.cuda.device_count()
    print(f"Total available GPUs: {num_gpus}")
    print("the prior knowledge is", config["prior_knowledge"], flush=True)
    S_I = gan_nerual.Generator(input_dim=400, output_dim=(2 * config["freq_numbers"] + 1) * 2).to(device)
    S_Omega = gan_nerual.omega_generator(input_dim=400, output_dim=2 * config["freq_numbers"]).to(device)
    # dp the train
    S_Omega = torch.nn.DataParallel(S_Omega, device_ids=[0])
    S_I = torch.nn.DataParallel(S_I, device_ids=[0])
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


    train(S_I, S_Omega, config["S_Omega_epoch"], config["S_I_epoch"])






