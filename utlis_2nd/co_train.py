import utlis_2nd.utlis_funcs as uf

import torch
import torch.optim as optim
import os

import numpy as np
import time
import torch.nn as nn
from geomloss import SamplesLoss
import utlis_2nd.neural_base_class as nn_base
class train_init():
    device = "cuda"
    def __init__(self,S_I,S_Omega,config,train_loader,valid_loader,test_loader,S_I_writer,S_Omega_writer):
        self.config=config
        self.train_loader=train_loader
        self.valid_loader=valid_loader
        self.test_loader=test_loader

        #criterion

        self.criterion_inference = nn.MSELoss()
        self.criterion_ini = nn.MSELoss()
        self.criterion_grad = nn.MSELoss()
        self.criterion_fourier =  nn.KLDivLoss(reduction='batchmean')


        # loss list

        self.g_omega_freq_loss_list = []
        self.inference_loss_list = []
        self.ini_loss_list = []
        self.grad_loss_list = []
        self.fourier_loss_list = []
        self.mse_loss_list = []

        #size

        self.train_size=self.config["train_size"]
        self.valid_size=self.config["valid_size"]

        # prepare:data_t

        self.numpy_t = np.linspace(0, 2, 100)
        self.numpy_t = self.numpy_t.reshape(100, 1)
        self.data_t = torch.from_numpy(self.numpy_t).float().to(self.device)
        self.data_t = self.data_t.unsqueeze(0).expand(32, -1, 1)

        #writer

        self.S_I_Writer=S_I_writer
        self.S_Omega_Writer=S_Omega_writer
        self.S_I =S_I
        self.S_Omega =S_Omega
        self.S_I_optimizer = optim.Adam(S_I.parameters(), lr=config["S_I_lr"])
        self.S_Omega_optimizer = optim.Adam(S_Omega.parameters(), lr=config["S_Omega_lr"])
        self.I_num_epoch = self.config["Inference_num_epoch"]
        self.Omega_num_epoch = self.config["Omega_num_epoch"]
        self.beta = self.config["beta"]
        self.freq_numbers = self.config["freq_numbers"]

    def train_inference_neural(self,process_name="train_process"):
        '''
        train the inference neural network
        return eval_value
        '''
        print("start train_inference")

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
                real_condition, _ = nn_base.convert_data(real, self.data_t, label, step=S_I_step)

                # omega_value
                with torch.no_grad():
                    generator_freq = self.S_Omega(real_condition)

                # generator_freq[batch,numbers,2]
                generator_freq = generator_freq.reshape(-1, self.config["freq_numbers"], 2)

                # compare the fourier domain's difference
                real_freq = nn_base.compute_spectrum(real,
                                                        beta=self.config["beta"],
                                                        domin_number=self.config["train_nomin"],
                                                        freq_number=self.config["freq_numbers"],
                                                        train_step=S_I_step,
                                                        filepath=self.S_I_Writer["train_analysis_file"],
                                                        name="real_data")

                # inference loss
                pred_coeffs = self.S_I(real_condition)
                pred_coeffs = pred_coeffs.reshape(self.config["batch_size"], -1, 2)

                if S_I_step == 1:
                    left_matrix, symbol_matrix = nn_base.Get_basis_function_info(self.config['prior_knowledge'])

                updated_symbol_list_z1, left_matrix, pred_data, pred_condition = nn_base.return_torch_version_matrix(
                    pred_coeffs,
                    generator_freq,
                    symbol_matrix)
                pred_freq = nn_base.compute_spectrum(pred_data,
                                                        beta=self.config["beta"],
                                                        freq_number=self.config["freq_numbers"],
                                                        train_step=S_I_step,
                                                        filepath=self.S_I_Writer["train_analysis_file"],
                                                        name="pred_data")

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
            self.S_I_Writer[process_name].add_scalars(process_name,
                                                        {
                                                        "inference_loss": epoch_inference_loss,
                                                        "fourier_loss": epoch_fourier_loss,
                                                        "mse_loss": epoch_mse_loss}, epoch)
            self.S_I_Writer[process_name].add_scalars(process_name+"_not for the loss",
                                                            {
                                                              "ini_loss": epoch_ini_loss,
                                                              "grad_loss": epoch_grad_loss}, epoch)

            self.inference_loss_list = []
            self.fourier_loss_list = []
            self.ini_loss_list = []
            self.grad_loss_list = []


            print(f"loss{epoch_inference_loss.item()}" + f"_epoch{epoch}" + "epoch_time" + \
                  f"{final_time - start_time}", flush=True)
            # #every 100epoch save the checkpoint
            if epoch % 100 == 0:
                checkpoint = {
                    "epoch": epoch,
                    "S_I_model_state_dict": self.S_I.module.state_dict(),
                    'S_Omega_optimizer_state_dict': self.S_I_optimizer.state_dict(),
                    "loss": epoch_inference_loss,
                }
                torch.save(checkpoint, self.S_I_Writer["model_checkpoint_path"])
            with torch.no_grad():
                eval_value=self.eval_inference_model(eval_data=self.valid_loader, eval_epoch=epoch,name="valid_process")
        return eval_value


    def train_omega_neural(self,process_name="train_process"):
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
                real_condition, _ = nn_base.convert_data(real, self.data_t, label, step=S_Omega_step)

                # omega_value
                pred_freq = self.S_Omega(real_condition)


                # generator_freq[batch,numbers,seq,2]
                pred_freq = pred_freq.reshape(-1, self.config["freq_numbers"], 2)

                # compare the fourier domain's difference
                real_freq,freq_index = nn_base.compute_spectrum_normlized(real,
                                                        domin_number=self.config["train_nomin"],
                                                        train_step=S_Omega_step,
                                                        filepath=self.S_Omega_Writer["train_analysis_file"],
                                                        name="real_data",label_save=label)

                # g_omega_loss kl divergence
                g_omega_freq_loss = self.criterion_fourier(pred_freq.log(), real_freq)


                self.g_omega_freq_loss_list.append(g_omega_freq_loss)

                # save for the data
                # optimizer
                self.S_Omega_optimizer.zero_grad()
                g_omega_freq_loss.backward()
                self.S_Omega_optimizer.step()

            final_time = time.time()
            epoch_g_omega_freq_loss = sum(self.g_omega_freq_loss_list) / len(self.g_omega_freq_loss_list)
            print("***epoch",epoch)
            self.S_Omega_Writer[process_name].add_scalar(process_name, epoch_g_omega_freq_loss, epoch)
            self.g_omega_freq_loss_list = []
            print(f"loss{epoch_g_omega_freq_loss.item()}" + f"_epoch{epoch}---" + "epoch_time" + \
                  f"{final_time - start_time}", flush=True)
            # #every 100 epoch save the checkpoint
            if epoch % 100 == 0:
                checkpoint = {
                    "epoch": epoch,
                    "S_Omega_model_state_dict": self.S_Omega.module.state_dict(),
                    'S_Omega_optimizer_state_dict': self.S_Omega_optimizer.state_dict(),
                    "loss": epoch_g_omega_freq_loss,
                }
                torch.save(checkpoint, self.S_Omega_Writer["model_checkpoint_path"])

            with torch.no_grad():
                eval_mse_value,u_stat,eval_mae=self.eval_omega_model(eval_data=self.valid_loader,
                                 eval_epoch=epoch,name="valid_process")
        return eval_mse_value,u_stat,eval_mae



    def eval_inference_model(self, eval_data,eval_epoch,name="valid_process"):
        '''

        :return: value of eval dict
        '''
        eval_sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05)
        device = "cuda"
        analysis_name=0
        if(name=="valid_process"):
            analysis_name="valid_analysis_file"
        elif(name=="test_process"):
            analysis_name="test_analysis_file"
        S_I_eval_step = 0
        criterion_fourier = nn.functional.kl_div(reduction="batchmean")
        eval_freq_loss_list = []
        eval_data_loss_list = []
        eval_grad_loss_list = []
        eval_u_stat_freq_list = []
        eval_u_stat_data_list = []
        eval_u_stat_grad_list = []
        eval_sinkhorn_data_list = []
        eval_sinkhorn_grad_list = []
        eval_sinkhorn_fouier_list = []
        eval_mae_list=[]
        for i, (batch_data, label) in enumerate(eval_data):
            S_I_eval_step += 1
            # get the condition_data and data_t and dict_str_solu
            # real_data
            real = batch_data[:, :, 7:9].to(device)
            label= label.to(device)
            # real_condition[batch,200,2]
            real_condition, _ = nn_base.convert_data(real, self.data_t, label, step=S_I_eval_step)
            if S_I_eval_step == 1:
                left_matrix, symbol_matrix = nn_base.Get_basis_function_info(self.config['prior_knowledge'])
            # omega_value
            with torch.no_grad():
                generator_freq = self.S_Omega(real_condition)

            # generator_freq[batch,numbers,2]
            generator_freq = generator_freq.reshape(-1, self.config["freq_numbers"], 2)

            # inference loss
            pred_coeffs = self.S_I(real_condition)
            pred_coeffs = pred_coeffs.reshape(-1, (2 * self.config["freq_numbers"] + 1), 2)
            updated_symbol_list, left_matrix, pred_data, pred_condition = nn_base.return_torch_version_matrix(
                pred_coeffs,
                generator_freq,
                symbol_matrix
            )
            real_freq = nn_base.compute_spectrum(real,
                                                    beta=self.config["beta"],
                                                    domin_number=self.config["valid_nomin"],
                                                    freq_number=self.config["freq_numbers"],
                                                    train_step=S_I_eval_step,
                                                    filepath=self.S_I_Writer[analysis_name],
                                                    name=name+"real_data",label_save=label)
            pred_freq = nn_base.compute_spectrum(pred_data,
                                                    beta=self.config["beta"],
                                                    domin_number=self.config["valid_nomin"],
                                                    freq_number=self.config["freq_numbers"],
                                                    train_step=S_I_eval_step,
                                                    filepath=self.S_I_Writer[analysis_name],
                                                    name=name+"pred_data",label_save=label)

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
            # [batch]-numbers sinkhorn distance
            data_sinkhorn=torch.mean(eval_sinkhorn_loss(real,pred_data))
            eval_sinkhorn_data_list.append(data_sinkhorn)

            grad_sinkhorn=torch.mean(eval_sinkhorn_loss(real_condition[:,100:,:],pred_condition[:,100:,:]))
            eval_sinkhorn_grad_list.append(grad_sinkhorn)

            fourier_sinkhorn=torch.mean(eval_sinkhorn_loss(real_freq,pred_freq))
            eval_sinkhorn_fouier_list.append(fourier_sinkhorn)

            #eval mae data
            eval_mae_list.append(nn.L1Loss()(pred_data,real))




        eval_freq_loss = sum(eval_freq_loss_list) / len(eval_freq_loss_list)
        eval_data_loss = sum(eval_data_loss_list) / len(eval_data_loss_list)
        eval_grad_loss = sum(eval_grad_loss_list) / len(eval_grad_loss_list)
        eval_u_stat_freq = sum(eval_u_stat_freq_list) / len(eval_u_stat_freq_list)
        eval_u_stat_data = sum(eval_u_stat_data_list) / len(eval_u_stat_data_list)
        eval_u_stat_grad = sum(eval_u_stat_grad_list) / len(eval_u_stat_grad_list)
        eval_sinkhorn_data=sum(eval_sinkhorn_data_list)/len(eval_sinkhorn_data_list)
        eval_sinkhorn_grad=sum(eval_sinkhorn_grad_list)/len(eval_sinkhorn_grad_list)
        eval_sinkhorn_fouier=sum(eval_sinkhorn_fouier_list)/len(eval_sinkhorn_fouier_list)
        eval_mae=sum(eval_mae_list)/len(eval_mae_list)

        # save the eval result
        self.S_I_Writer[name].add_scalars(name, { 'eval_freq_loss': eval_freq_loss,
                                                   'eval_data_loss': eval_data_loss,
                                                   'eval_grad_loss': eval_grad_loss,
                                                   'eval_u_stat_freq': eval_u_stat_freq,
                                                   'eval_u_stat_data': eval_u_stat_data,
                                                   'eval_u_stat_grad': eval_u_stat_grad,
                                                    'eval_sinkhorn_data':eval_sinkhorn_data,
                                                    'eval_sinkhorn_grad':eval_sinkhorn_grad,
                                                    'eval_sinkhorn_fouier':eval_sinkhorn_fouier,
                                                    'eval_mae':eval_mae
                                                   }, eval_epoch)

        return_dict= {                             'eval_freq_loss': eval_freq_loss,
                                                   'eval_data_loss': eval_data_loss,
                                                   'eval_grad_loss': eval_grad_loss,
                                                   'eval_u_stat_freq': eval_u_stat_freq,
                                                   'eval_u_stat_data': eval_u_stat_data,
                                                   'eval_u_stat_grad': eval_u_stat_grad,
                                                    'eval_sinkhorn_data':eval_sinkhorn_data,
                                                    'eval_sinkhorn_grad':eval_sinkhorn_grad,
                                                    'eval_sinkhorn_fouier':eval_sinkhorn_fouier,
                                                    'eval_mae':eval_mae
                                                   }
        return return_dict

    def eval_omega_model(self,eval_data,eval_epoch,name="valid_process"):
        '''
         :param eval_data: eval data
          record mse and u-stat
          return:mse
        '''
        analysis_name=0
        if(name=="valid_process"):
            analysis_name="valid_analysis_file"
        elif(name=="test_process"):
            analysis_name="test_analysis_file"

        eval_step = 0
        device = "cuda"
        criterion_fourier = nn.MSELoss()
        eval_mse_loss_list = []
        eval_u_stat_list = []
        eval_mae_loss_list=[]

        for i, (batch_data, label) in enumerate(eval_data):
            eval_step = eval_step + 1
            # get the condition_data and data_t and dict_str_solu

            # real_data
            real = batch_data[:, :, 7:9].to(device)

            label= label.to(device)

            # real_condition[batch,200,2]
            real_condition, _ = nn_base.convert_data(real,self.data_t, label, step=eval_step)

            # omega_value
            pred_freq = self.S_Omega(real_condition)

            # generator_freq[batch,numbers,2]
            pred_freq = pred_freq.reshape(-1, self.config["freq_numbers"], 2)

            if (eval_step % self.config["valid_nomin"] ==0):
                epoch_omega = eval_step / self.config["valid_nomin"]
                info = {"epoch_omega": epoch_omega,
                        "pred_freq": pred_freq,
                        }

                torch.save(info, self.S_Omega_Writer[analysis_name] +
                           "/" + "pred_freq" + "_" + f"{int(epoch_omega)}.pth")

            # compare the fourier domain's difference

            real_freq = nn_base.compute_spectrum(real,
                                                    beta=self.config["beta"],
                                                    domin_number=self.config["valid_nomin"],
                                                    freq_number=self.config["freq_numbers"],
                                                    train_step=eval_step,
                                                    filepath=self.S_Omega_Writer[analysis_name],
                                                    name=name+"real_data",label_save=label)

            # mse
            g_omega_freq_loss = criterion_fourier(pred_freq, real_freq)
            eval_mse_loss_list.append(g_omega_freq_loss)
            # u_stat
            u_stat = uf.theil_u_statistic(pred_freq, real_freq)
            eval_u_stat_list.append(u_stat)
            #mae -freq

            eval_mae_loss_list.append(nn.L1Loss()(pred_freq,real_freq))

        eval_mse_loss = sum(eval_mse_loss_list) / len(eval_mse_loss_list)
        eval_u = sum(eval_u_stat_list) / len(eval_u_stat_list)
        eval_mae= sum(eval_mae_loss_list)/len(eval_mae_loss_list)


        self.S_Omega_Writer[name].add_scalars(name, {
                                                "mse_loss": eval_mse_loss,
                                                "u_stat": eval_u,
                                                "mae_loss":eval_mae,
                                                }, eval_epoch)
        return eval_mse_loss,eval_u,eval_mae
