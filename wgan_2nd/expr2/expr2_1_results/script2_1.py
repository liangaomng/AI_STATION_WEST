#this is 4 expr2

import os
import torch
torch.cuda.empty_cache()
from collections import namedtuple
import utlis_2nd.cusdom as custom
import utlis_2nd.super_learn_task_expr2 as expr2
import warnings

###
folder_num=1 ###start1 1
#about train step and regularization
expr2.config["Inference_num_epoch"]=1000
###
def handle_path(read_abso_path,yaml_path,result_path):
   # read_abso_path="E:\expr\AI_STATION_WEST-master\wgan_2nd\complex_center_dataset\combined_data.pt"
    ###
    warnings.filterwarnings("ignore")
    #data description
    Data_description=namedtuple("Data_description",["t_steps",
                                                    "vesting_s",
                                                    "vari_numb",
                                                    "sample_rate",
                                                    "freq_numb"])
    #temp info
    Soft_arg_temp = namedtuple("Soft_arg_temp", ["learnable", "value"])
    Gumble_temp = namedtuple("Gumble_temp", ["learnable", "value"])
    Sample_info = namedtuple("Sample_info", ["type", "sample_numb"])
    #loss regularization
    Sinkhorn_loss_info = namedtuple("Sinkhorn_loss_info", ["loss_en", "p","blur"])
    Fourier_loss_info = namedtuple("Fourier_loss_info", ["loss_en","labmbda_fourier"])
    Lasso_loss_info = namedtuple("Lasso_loss_info", ["loss_en", "labmbda_lasso"])

    # split data & get the t_steps
    train_loader,valid_loader,test_loader,yaml_config =\
        custom.return_train_valid_test4loader(abso_path=read_abso_path,yaml_path=yaml_path)
    # look（shape）
    for batch_idx, (data, target) in enumerate(train_loader):
        print("shape:", data.shape)
        print("label for csv:",target.shape)
        break

    soft_arg_temp = Soft_arg_temp(
                                    learnable=yaml_config["Soft_argmax_info"][0],
                                    value=yaml_config["Soft_argmax_info"][1])
    sample_info = Sample_info(
                                type=yaml_config["Sample_info"][0],
                                sample_numb=yaml_config["Sample_info"][1])

    expr2.config["train_nomin"]= int((yaml_config['train_size_percent']*yaml_config['all_solus_numbers'])/yaml_config["batch_size"])
    expr2.config["valid_nomin"]= int((yaml_config['valid_size_percent']*yaml_config['all_solus_numbers'])/yaml_config["batch_size"])
    expr2.config["test_nomin"]= int((yaml_config['test_size_percent']*yaml_config['all_solus_numbers'])/yaml_config["batch_size"])
    expr2.config["train_loader"]= train_loader
    expr2.config["valid_loader"]= valid_loader
    expr2.config["test_loader"]= test_loader
    expr2.config["device"] = "cuda"
    expr2.config["data_description"]=yaml_config["data_description"]
    expr2.config["hidden_act"]= "rational"
    expr2.config["SI_lr"]= 1e-5
    expr2.config["inference_output_act"]="Identity"
    expr2.config["vari_number"]= yaml_config["vari_number"]
    expr2.config["sample_vesting"]=2 #2s
    #about temp
    expr2.config["soft_arg_info"] = soft_arg_temp
    expr2.config["gumble_info"] = None
    expr2.config["sample_info"] = sample_info
    #about regularization
    expr2.config["Sinkhorn_loss_info"] = yaml_config["Sinkhorn_loss_info"]
    expr2.config["Fourier_loss_info"] = yaml_config["Fourier_loss_info"]
    expr2.config["Lasso_loss_info"] = yaml_config["Lasso_loss_info"]

    expr2.do_expr(
        results_save_path=result_path,
        folder_num=foler_num,
        train_config=expr2.config,
        model_type="inference_net")


import argparse


if __name__=="__main__":
    #count the yaml and do the expr
    #find the dir' yaml
    parser=argparse.ArgumentParser()
    parser.add_argument("--results_path",type=str,default="E:\expr\AI_STATION_WEST-master\wgan_2nd\expr2\expr2_1_results\expr2_1_")
    parser.add_argument("--read_abso_path",type=str,default="E:\expr\AI_STATION_WEST-master\wgan_2nd\complex_center_dataset\combined_data.pt")
    args=parser.parse_args()

    #do the expr
    for foler_num in [1,2,3]:
        # return test' results
        yaml_path=f".\yaml\expr2_1_{foler_num}.yaml"
        handle_path(read_abso_path=args.read_abso_path, yaml_path=yaml_path,result_path=args.results_path)
