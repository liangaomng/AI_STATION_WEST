import numpy as np

import utlis_2nd.utlis_funcs as uf
import torch
import torch.optim as optim
import time
import torch.nn as nn
from geomloss import SamplesLoss
import matplotlib.pyplot as plt
class train_init():
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

    def train_inference_neural(self,
                               process_name="train_process",
                               device="cuda",
                               save_2visualfig=True):
        '''
        train the inference neural network
        return eval_value
        '''
        print("start train_inference")
        S_I_step = 0

        #
        for epoch in range(self.I_num_epoch):

            start_time = time.time()
            for i, (batch_data, label) in enumerate(self.train_loader):
                S_I_step += 1
                # get the condition_data and data_t and dict_str_solu

                # real_data
                real = batch_data[:, :, 7:9].to(device)

                # omega_value
                # with torch.no_grad():
                #     self.S_Omega.eval()
                        #pretain_freq_distrubtion = self.S_Omega(real) #[batch,freq_index,2]

                # compare the fourier domain's difference
                real_freq_distrubtion  = self.S_I.return_fft_spectrum(real,need_norm=True)

                # inference loss
                pred_coeffs = self.S_I(real) #[batch,freq_index*2,2]
                #print("pred_coeff",pred_coeffs.shape) #[batch,freq_index*1,2]
                #
                # print("pretain_freq_distrubtion", pretain_freq_distrubtion.shape)#[batch,freq_index*1,2]
                #
                # #plot the freq distrubtion



                left_matrix,pred_data=self.S_I.return_pred_data(pred_coeffs,real_freq_distrubtion)
                pred_freq_distrubtion = self.S_I.return_fft_spectrum(pred_data,need_norm=True)

                if(save_2visualfig==True):
                    #plot the freq distribution and the
                    fig,ax=plt.subplots(2,1)
                    ax[0].plot(np.linspace(0,50,51),pred_freq_distrubtion[0, :, 0].cpu().detach().numpy(), label="pred_freq_distrubtion")
                    ax[0].plot(np.linspace(0,50,51),real_freq_distrubtion[0, :, 0].cpu().detach().numpy(), label="real_freq_distrubtion")
                    ax[1].plot(np.linspace(0,2,100),pred_data[0, :, 0].cpu().detach().numpy(), label="pred_data")
                    ax[0].legend()
                    ax[1].legend()



                # save for the data-visualization
                mse_loss = self.criterion_inference(pred_data, real)
                self.mse_loss_list.append(mse_loss)
                fouier_loss = self.criterion_fourier(pred_freq_distrubtion, real_freq_distrubtion)
                self.fourier_loss_list.append(fouier_loss)

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
            epoch_mse_loss = sum(self.mse_loss_list) / len(self.mse_loss_list)
            self.S_I_Writer[process_name].add_scalars(process_name,
                                                        {
                                                        "inference_loss": epoch_inference_loss,
                                                        "fourier_loss": epoch_fourier_loss,
                                                        "mse_loss": epoch_mse_loss}, epoch)


            self.inference_loss_list = []
            self.fourier_loss_list = []


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
                torch.save(checkpoint, self.S_I_Writer["model_checkpoint_path"])
            with torch.no_grad():
                eval_value=self.eval_inference_model(eval_data=self.valid_loader, eval_epoch=epoch,name="valid_process")
        return eval_value


    def train_omega_neural(self,
                           process_name="train_process",
                           device="cuda"):
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

                # real_data [batch,seq,2]
                real = batch_data[:, :, 7:9].to(device)
                label = label


                # return [batch,freq_index,2]
                pred_freq = self.S_Omega(real)

                # compare the real fourier domain's difference
                #*** problem  if datapara :self.S_Omega.module.return_fft_spectrum(
                real_freq= self.S_Omega.return_fft_spectrum(real,need_norm=True)

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
                    "S_Omega_model_state_dict": self.S_Omega.state_dict(),
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
        analysis_name=0
        if(name=="valid_process"):
            analysis_name="valid_analysis_file"
        elif(name=="test_process"):
            analysis_name="test_analysis_file"
        S_I_eval_step = 0

        eval_freq_loss_list = []
        eval_data_loss_list = []
        eval_u_stat_data_list = []
        eval_sinkhorn_data_list = []
        eval_mae_list=[]
        for i, (batch_data, label) in enumerate(eval_data):
            S_I_eval_step += 1
            # get the condition_data and data_t and dict_str_solu
            # real_data
            real = batch_data[:, :, 7:9]

            # omega_value
            with torch.no_grad():
                self.S_Omega.eval()
                pretain_freq_distrubtion = self.S_Omega(real)  # [batch,freq_index,2]

            # compare the fourier domain's difference
            real_freq_distrubtion = self.S_I.return_fft_spectrum(real, need_norm=True)

            # inference loss
            pred_coeffs = self.S_I(real)  # [batch,freq_index*2,2]
            left_matrix,pred_data = self.S_I.return_pred_data(pred_coeffs, pretain_freq_distrubtion)

            pred_freq_distrubtion = self.S_I.return_fft_spectrum(pred_data, need_norm=True)

            # g_omega_loss
            g_omega_freq_loss = self.criterion_fourier(pred_freq_distrubtion, real_freq_distrubtion)
            eval_freq_loss_list.append(g_omega_freq_loss)
            # data_loss
            data_loss = self.criterion_inference(pred_data, real)
            eval_data_loss_list.append(data_loss)


            # u_stat
            u_stat_data = uf.theil_u_statistic(pred_data, real)
            eval_u_stat_data_list.append(u_stat_data)
            # [batch]-numbers sinkhorn distance
            data_sinkhorn=torch.mean(eval_sinkhorn_loss(real,pred_data))
            eval_sinkhorn_data_list.append(data_sinkhorn)
            #eval mae data
            eval_mae_list.append(nn.L1Loss()(pred_data,real))




        eval_freq_loss = sum(eval_freq_loss_list) / len(eval_freq_loss_list)
        eval_data_loss = sum(eval_data_loss_list) / len(eval_data_loss_list)

        eval_u_stat_data = sum(eval_u_stat_data_list) / len(eval_u_stat_data_list)
        eval_sinkhorn_data=sum(eval_sinkhorn_data_list)/len(eval_sinkhorn_data_list)
        eval_mae=sum(eval_mae_list)/len(eval_mae_list)

        # save the eval result
        self.S_I_Writer[name].add_scalars(name, { 'eval_freq_loss': eval_freq_loss,
                                                   'eval_data_loss': eval_data_loss,
                                                   'eval_u_stat_data': eval_u_stat_data,
                                                    'eval_sinkhorn_data':eval_sinkhorn_data,
                                                    'eval_mae':eval_mae
                                                   }, eval_epoch)

        return_dict= {
                                                    'eval_freq_loss': eval_freq_loss,
                                                   'eval_data_loss': eval_data_loss,
                                                   'eval_u_stat_data': eval_u_stat_data,
                                                    'eval_sinkhorn_data':eval_sinkhorn_data,
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


        eval_mse_loss_list = []
        eval_u_stat_list = []
        eval_mae_loss_list=[]

        for i, (batch_data, label) in enumerate(eval_data):
            eval_step = eval_step + 1
            # get the condition_data and data_t and dict_str_solu

            # real_data
            real = batch_data[:, :, 7:9]


            # omega_value
            pred_freq = self.S_Omega(real)

            # compare the real fourier domain's difference
            real_freq = self.S_Omega.return_fft_spectrum(real, need_norm=True)

            # g_omega_loss kl divergence
            g_omega_freq_loss = self.criterion_fourier(pred_freq.log(), real_freq)

            self.g_omega_freq_loss_list.append(g_omega_freq_loss)

            if (eval_step % self.config["valid_nomin"] ==0):
                epoch_omega = eval_step / self.config["valid_nomin"]
                info = {"epoch_omega": epoch_omega,
                        "pred_freq": pred_freq,
                        }

                torch.save(info, self.S_Omega_Writer[analysis_name] +
                           "/" + "pred_freq" + "_" + f"{int(epoch_omega)}.pth")


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
