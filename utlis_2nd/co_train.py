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

class train_init():
    device = "cuda"
    def __init__(self,S_I,S_Omega,config,train_loader,valid_loader,test_loader,writer):
        self.config=config
        self.train_loader=train_loader
        self.valid_loader=valid_loader
        self.test_loader=test_loader

        self.criterion_inference = nn.MSELoss()
        self.criterion_ini = nn.MSELoss()
        self.criterion_grad = nn.MSELoss()
        self.criterion_fourier = nn.MSELoss()
        # loss list
        self.g_omega_freq_loss_list = []
        self.inference_loss_list = []
        self.ini_loss_list = []
        self.grad_loss_list = []
        self.fourier_loss_list = []
        self.mse_loss_list = []
        # prepare:data_t
        self.numpy_t = np.linspace(0, 2, 100)
        self.numpy_t = self.numpy_t.reshape(100, 1)
        self.data_t = torch.from_numpy(self.numpy_t).float().to(self.device)
        self.data_t = self.data_t.unsqueeze(0).expand(32, -1, 1)
        #writer
        self.writer=writer
        self.S_I =S_I
        self.S_Omega =S_Omega
        self.S_I_optimizer = optim.Adam(S_I.parameters(), lr=config["S_I_lr"])
        self.S_Omega_optimizer = optim.Adam(S_Omega.parameters(), lr=config["S_Omega_lr"])
        self.I_num_epoch = self.config["Inference_num_epoch"]
        self.Omega_num_epoch = self.config["Omega_num_epoch"]
        self.beta = self.config["beta"]
        self.freq_numbers = self.config["freq_numbers"]

    def train_inference_neural(self,*args):
        '''
        train the inference neural network
        return eval_value
        '''
        print("train_inference")

        S_I_step = 0
        device = "cuda"
        #
        for epoch in range(self.I_num_epoch):
            start_time = time.time()
            for i, (batch_data, label) in enumerate(self.train_loader):
                S_I_step += 1
                # get the condition_data and data_t and dict_str_solu
                # real_data
                real = batch_data[:, :, 7:9].to(device)
                # real_condition[batch,200,2]
                real_condition, _ = gan_nerual.convert_data(real, self.data_t, label, step=S_I_step)

                # omega_value
                with torch.no_grad():
                    generator_freq = self.S_Omega(real_condition)

                # generator_freq[batch,numbers,2]
                generator_freq = generator_freq.reshape(-1, self.config["freq_numbers"], 2)

                # compare the fourier domain's difference
                real_freq = gan_nerual.compute_spectrum(real,
                                                        beta=self.config["beta"],
                                                        freq_number=self.config["freq_numbers"],
                                                        train_step=S_I_step,
                                                        filepath=self.config["save_plot_tb_path"] + self.config[
                                                            "training_data_path"] +self. config["training_inference_path"],
                                                        name="real")

                # inference loss
                pred_coeffs = self.S_I(real_condition)
                pred_coeffs = pred_coeffs.reshape(self.config["batch_size"], -1, 2)

                if S_I_step == 1:
                    left_matrix, symbol_matrix = gan_nerual.Get_basis_function_info(self.config['prior_knowledge'])

                updated_symbol_list_z1, left_matrix, pred_data, pred_condition = gan_nerual.return_torch_version_matrix(
                    pred_coeffs,
                    generator_freq,
                    symbol_matrix)
                pred_freq = gan_nerual.compute_spectrum(pred_data,
                                                        beta=self.config["beta"],
                                                        freq_number=self.config["freq_numbers"],
                                                        train_step=S_I_step,
                                                        filepath=self.config["save_plot_tb_path"] + self.config[
                                                            "training_data_path"] + self.config["training_inference_path"],
                                                        name="pred")

                # save for the data---PINN
                mse_loss = self.criterion_inference(pred_data, real)
                self.mse_loss_list.append(mse_loss)
                fouier_loss = self.criterion_fourier(pred_freq, real_freq)
                self.fourier_loss_list.append(fouier_loss)
                # just for record
                gradient_loss = self.criterion_grad(pred_condition[:, 100:, :], real_condition[:, 100:, :])
                self.grad_loss_list.append(gradient_loss)
                ini_loss = self.criterion_ini(pred_condition[:, 0, :], real_condition[:, 0, :])
                self.ini_loss_list.append(ini_loss)

                infer_loss = mse_loss + self.config["lamba_fourier"] * fouier_loss
                self.inference_loss_list.append(infer_loss)
                # optimizer
                self.S_I_optimizer.zero_grad()
                # loss
                infer_loss.backward()
                self.S_I_optimizer.step()

            final_time = time.time()

            epoch_inference_loss = sum(self.inference_loss_list) / len(self.inference_loss_list)
            epoch_fourier_loss = sum(self.fourier_loss_list) / len(self.fourier_loss_list)
            epoch_ini_loss = sum(self.ini_loss_list) / len(self.ini_loss_list)
            epoch_grad_loss = sum(self.grad_loss_list) / len(self.grad_loss_list)
            epoch_mse_loss = sum(self.mse_loss_list) / len(self.mse_loss_list)
            self.writer.add_scalars("trian_inference_loss", {"inference_loss": epoch_inference_loss,
                                                        "fourier_loss": epoch_fourier_loss,
                                                        "mse_loss": epoch_mse_loss}, epoch)
            self.writer.add_scalars("trian_inference_record_not_loss", {"ini_loss": epoch_ini_loss,
                                                                   "grad_loss": epoch_grad_loss}, epoch)

            self.inference_loss_list = []
            self.fourier_loss_list = []
            self. ini_loss_list = []
            self.grad_loss_list = []
            vmse_loss_list = []

            print(f"loss{epoch_inference_loss.item()}" + f"_epoch{epoch}" + "epoch_time" + \
                  f"{final_time - start_time}", flush=True)
            # #every 100epoch save the checkpoint
            if epoch % 100 == 0:
                checkpoint = {
                    "epoch": epoch,
                    "S_I_model_state_dict": self.S_I.state_dict(),
                    'S_Omega_optimizer_state_dict': self.S_I_optimizer.state_dict(),
                    "loss": epoch_inference_loss,
                }
                torch.save(checkpoint, self.config["save_plot_tb_path"] + '/Inference_checkpoint.pth')
            with torch.no_grad():
                eval_value=self.eval_inference_model(eval_data=self.valid_loader, eval_epoch=epoch)
        return eval_value


    def train_omega_neural(self):
        # train the omega neural network
        '''
        return:value of mse
        '''

        print("start train the omega neural network")
        S_Omega_step = 0
        for epoch in range(self.Omega_num_epoch):
            start_time = time.time()
            for i, (batch_data, label) in enumerate(self.train_loader):
                S_Omega_step += 1

                # get the condition_data and data_t and dict_str_solu

                # real_data
                real = batch_data[:, :, 7:9].to("cuda")
                label = label.to("cuda")

                # real_condition[batch,200,2]
                real_condition, _ = gan_nerual.convert_data(real, self.data_t, label, step=S_Omega_step)

                # omega_value
                generator_freq = self.S_Omega(real_condition)

                # generator_freq[batch,numbers,2]
                generator_freq = generator_freq.reshape(-1, self.config["freq_numbers"], 2)

                # compare the fourier domain's difference
                real_freq = gan_nerual.compute_spectrum(real,
                                                        beta=self.config["beta"],
                                                        freq_number=self.config["freq_numbers"],
                                                        train_step=S_Omega_step,
                                                        filepath=self.config["save_plot_tb_path"]\
                                                             + self.config["training_data_path"]  \
                                                             + self.config["training_omega_data_path"],
                                                        name="real")
                # g_omega_loss
                g_omega_freq_loss = self.criterion_fourier(generator_freq, real_freq)
                self.g_omega_freq_loss_list.append(g_omega_freq_loss)

                # save for the data
                # optimizer
                self.S_Omega_optimizer.zero_grad()
                g_omega_freq_loss.backward()
                self.S_Omega_optimizer.step()

            final_time = time.time()
            epoch_g_omega_freq_loss = sum(self.g_omega_freq_loss_list) / len(self.g_omega_freq_loss_list)
            self.writer.add_scalar("train_omega_freq_loss", epoch_g_omega_freq_loss, epoch)
            self.g_omega_freq_loss_list = []
            print(f"loss{epoch_g_omega_freq_loss.item()}" + f"_epoch{epoch}---" + "epoch_time" + \
                  f"{final_time - start_time}", flush=True)
            # #every 100epoch save the checkpoint
            if epoch % 100 == 0:
                checkpoint = {
                    "epoch": epoch,
                    "S_Omega_model_state_dict": self.S_Omega.state_dict(),
                    'S_Omega_optimizer_state_dict': self.S_Omega_optimizer.state_dict(),
                    "loss": epoch_g_omega_freq_loss,
                }
                torch.save(checkpoint, self.config["save_plot_tb_path"] + '/omega_checkpoint.pth')
            with torch.no_grad():
                eval_mse_value,u_stat=self.eval_omega_model(eval_data=self.valid_loader,
                                 eval_epoch=epoch)
        return eval_mse_value,u_stat

    def eval_inference_model(self, eval_data,eval_epoch):
        '''

        :return: value of eval_data_loss' mse
        '''
        device = "cuda"
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
            real_condition, _ = gan_nerual.convert_data(real, self.data_t, label, step=S_I_eval_step)
            if S_I_eval_step == 1:
                left_matrix, symbol_matrix = gan_nerual.Get_basis_function_info(self.config['prior_knowledge'])
            # omega_value
            with torch.no_grad():
                generator_freq = self.S_Omega(real_condition)

            # generator_freq[batch,numbers,2]
            generator_freq = generator_freq.reshape(-1, self.config["freq_numbers"], 2)

            # inference loss
            pred_coeffs = self.S_I(real_condition)
            pred__coeffs = pred__coeffs.reshape(-1, (2 * self.config["freq_numbers"] + 1), 2)
            updated_symbol_list, left_matrix, pred_data, pred_condition = gan_nerual.return_torch_version_matrix(
                pred__coeffs,
                generator_freq,
                symbol_matrix
            )
            real_freq = gan_nerual.compute_spectrum(real,
                                                    beta=self.config["beta"],
                                                    freq_number=self.config["freq_numbers"],
                                                    train_step=S_I_eval_step,
                                                    filepath=self.config["save_plot_tb_path"] \
                                                         + self.config["eval_data_path"] \
                                                         + self.config["eval_inference_path"],
                                                    name="real")
            pred_freq = gan_nerual.compute_spectrum(pred_data,
                                                    beta=self.config["beta"],
                                                    freq_number=self.config["freq_numbers"],
                                                    train_step=S_I_eval_step,
                                                    filepath=self.config["save_plot_tb_path"] \
                                                         + self.config["eval_data_path"] \
                                                         + self.config["eval_inference_path"],
                                                    name="pred")

            # g_omega_loss
            g_omega_freq_loss = criterion_fourier(pred_freq, real_freq)
            eval_freq_loss_list.append(g_omega_freq_loss)
            # data_loss
            data_loss = criterion_fourier(pred_data, real)
            eval_data_loss_list.append(data_loss)
            # grad_loss
            grad_loss = criterion_fourier(pred_condition[:, 100:, :], real_condition[:, 100:, :])
            eval_grad_loss_list.append(grad_loss)
            # u_stat
            u_stat_freq = uf.theil_u_statistic(pred_freq, real_freq)
            u_stat_data = uf.theil_u_statistic(pred_data, real)
            u_stat_condition = uf.theil_u_statistic(pred_condition[:, 100:, :], real_condition[:, 100:, :])
            eval_u_stat_freq_list.append(u_stat_freq)
            eval_u_stat_data_list.append(u_stat_data)
            eval_u_stat_grad_list.append(u_stat_condition)
            # cost_matirx infinite_norm

        cost_matrix = uf.cost_matrix(real,pred_data)  # [batch,50,50,2]
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
        self.writer.add_scalars('eval_inference', {'eval_freq_loss': eval_freq_loss,
                                                   'eval_data_loss': eval_data_loss,
                                                   'eval_grad_loss': eval_grad_loss,
                                                   'eval_u_stat_freq': eval_u_stat_freq,
                                                   'eval_u_stat_data': eval_u_stat_data,
                                                   'eval_u_stat_grad': eval_u_stat_grad,
                                                   'eval_infinite_cost': eval_infinite_cost
                                                   }, eval_epoch)
        return eval_data_loss

    def eval_omega_model(self,eval_data,eval_epoch,name="eval_omega"):
        '''
         :param eval_data: eval data
          record mse and u-stat
          return:mse
        '''
        eval_step = 0
        device = "cuda"
        criterion_fourier = nn.MSELoss()
        eval_mse_loss_list = []
        eval_u_stat_list = []

        for i, (batch_data, label) in enumerate(eval_data):
            eval_step = eval_step + 1
            # get the condition_data and data_t and dict_str_solu

            # real_data
            real = batch_data[:, :, 7:9].to(device)

            # real_condition[batch,200,2]
            real_condition, _ = gan_nerual.convert_data(real,self.data_t, label, step=eval_step)

            # omega_value
            generator_freq = self.S_Omega(real_condition)

            # generator_freq[batch,numbers,2]
            generator_freq = generator_freq.reshape(-1, self.config["freq_numbers"], 2)
            # compare the fourier domain's difference
            real_freq = gan_nerual.compute_spectrum(real,
                                                    beta=self.config["beta"],
                                                    freq_number=self.config["freq_numbers"],
                                                    train_step=eval_step,
                                                    filepath=self.config["save_plot_tb_path"]\
                                                         +self.config["eval_data_path"] \
                                                         +self.config["eval_omega_data_path"],
                                                    name="real")

            # g_omega_loss
            g_omega_freq_loss = criterion_fourier(generator_freq, real_freq)
            eval_mse_loss_list.append(g_omega_freq_loss)
            # u_stat
            u_stat = uf.theil_u_statistic(generator_freq, real_freq)
            eval_u_stat_list.append(u_stat)

        eval_mse_loss = sum(eval_mse_loss_list) / len(eval_mse_loss_list)
        eval_u = sum(eval_u_stat_list) / len(eval_u_stat_list)

        self.writer.add_scalars(name, {"mse_loss": eval_mse_loss,
                                               "u_stat": eval_u,
                                                }, eval_epoch)
        return eval_mse_loss,eval_u
