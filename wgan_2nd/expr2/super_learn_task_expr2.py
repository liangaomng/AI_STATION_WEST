#co-train

import utlis_2nd.utlis_funcs as uf
import utlis_2nd.gan_nerual as gan_nerual
import utlis_2nd.co_train as co_train
from utlis_2nd.cusdom import *
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import warnings
import utlis_2nd.neural_base_class as nn_base

warnings.filterwarnings("ignore")
# set_default_dtype float64
torch.set_default_dtype(torch.float64)
from utlis_2nd.cusdom import train_loader,valid_loader,test_loader
#record expr_time：
from datetime import datetime
device = 'cuda'
omega_net_writer = {"train_process": 0,
                    "valid_process": 0,
                    "test_process": 0,
                    "model_checkpoint_path": 0,
                    "train_analysis_file": 0,
                    "valid_analysis_file": 0,
                    "test_analysis_file": 0}  # "train,valid,test,model_checkpoint"
inference_net_writer = {"train_process": 0,
                            "valid_process": 0,
                            "test_process": 0,
                            "model_checkpoint_path": 0,
                            "train_analysis_file": 0,
                            "valid_analysis_file": 0,
                            "test_analysis_file": 0}  # "train,valid,test,model_checkpoint"
#expr2********
expr_data_path_basis="/liangaoming/conda_lam/expriments/paper1/expr2/expr2_"
general_file_structure = {
    "train_process": expr_data_path_basis + "/train_process",
    "valid_process": expr_data_path_basis + "/valid_process",
    "test_process": expr_data_path_basis + "/test_process",
    "model_checkpoint_path": expr_data_path_basis + "/model_check_point",
    "CSV": expr_data_path_basis + "/csv",
    "tb_event": "/tb_event",
    "train_analysis_file": expr_data_path_basis + "/train_process",
    "valid_analysis_file": expr_data_path_basis + "/valid_process",
    "test_analysis_file": expr_data_path_basis + "/test_process",
}
config = {
    # basis
    "batch_size": 256,
    "g_neural_network_width": 512,
    "all_data_save_path": general_file_structure,
    "lamba_ini": 1,
    "lamba_grad": 1,
    "lamba_gp": 1,
    "lamba_fourier": 1,
    "prior_knowledge": {"basis_1": "x**0", "basis_2": "sin", "basis_3": "cos"},
    "S_I_lr": 1e-4,
    "S_Omega_lr": 1e-4,
    "grads_types": {"boundary": 3, "inner": 5},  # boundary:forward and backward;inner:five points
    "beta": 2,
    "freq_numbers": 1,
    # save path
    "expr_data_path": general_file_structure["train_process"],
    "CSV": general_file_structure["CSV"],
    "tb_event": general_file_structure["tb_event"],
    # writer
    "omega_net_writer": omega_net_writer,
    "inference_net_writer": inference_net_writer,
    "train_size": train_size,
    "valid_size": valid_size,
    "test_size": test_size,

    "train_nomin": train_size / batch__size,
    "valid_nomin": valid_size / batch__size,
    "test_nomin": test_size / batch__size,

    "seed": 42,
    "Omega_num_epoch": 5,
    "Inference_num_epoch": 5000,
    "co_train_time": 100,
    "infer_minimum": 1e-1,

    "train_loader": train_loader,
    "valid_loader": valid_loader,
    "test_loader": test_loader,
    "current_expr_start_time": 0,
    "omega_model_load_path":"/liangaoming/conda_lam/expriments/paper1/expr1/expr1_98_data/model_check_point/omega_net/omega_net_model.pth",

    "data_length":train_loader.dataset[0][0].shape[0]  # data length
}

def current2_time():
    current_time = datetime.now()
    current_start_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return current_start_time
def record_init(folder_num):
    # record expr_time：
    current_time=current2_time()
    expr_data_path = expr_data_path_basis+ f"{folder_num}"+"_data"
    # define the structure
    global general_file_structure
    general_file_structure ={
        "train_process": expr_data_path + "/train_process",
        "valid_process": expr_data_path + "/valid_process",
        "test_process": expr_data_path + "/test_process",
        "model_checkpoint_path": expr_data_path + "/model_check_point",
        "CSV": expr_data_path + "/csv",
        "tb_event": "/tb_event",
        "train_analysis_file": expr_data_path + "/train_process",
        "valid_analysis_file": expr_data_path + "/valid_process",
        "test_analysis_file": expr_data_path + "/test_process",
    }

    for key in omega_net_writer.keys():

        path = general_file_structure[key] + "/omega_net"

        if key == "model_checkpoint_path":
            omega_net_writer[key] = path + "/omega_net_model.pth"

        if key == "train_process" or key == "valid_process" or key == "test_process":
            path = path + general_file_structure["tb_event"]
            omega_net_writer[key] = SummaryWriter(path)

        if key == "train_analysis_file" or key == "valid_analysis_file" or key == "test_analysis_file":
            path = general_file_structure[key] + "/omega_net" + "/analysis_files"
            omega_net_writer[key] = path
        if key == "test_analysis_file":
            path = general_file_structure[key] + "/omega_net" + "/analysis_files"
            omega_net_writer[key] = path

    for key in inference_net_writer.keys():

        path=general_file_structure[key]+"/inference_net"

        if key=="model_checkpoint_path":
            inference_net_writer[key]=path+"/inference_net_model.pth"

        if key =="train_process" or key=="valid_process" or key=="test_process":
            path=path+general_file_structure["tb_event"]
            inference_net_writer[key] = SummaryWriter(path)
        if key =="train_analysis_file" or key=="valid_analysis_file" or key=="test_analysis_file":
            path=general_file_structure[key]+"/inference_net"+"/analysis_files"
            inference_net_writer[key]=path
        if key=="test_analysis_file":
            path=general_file_structure[key]+"/omega_net"+"/analysis_files"
            inference_net_writer[key]=path
    # config
    global  config
    #update
    config["expr_data_path"]=expr_data_path
    config["CSV"]=general_file_structure["CSV"]
    config["tb_event"]=general_file_structure["tb_event"]
    config["omega_net_writer"]=omega_net_writer
    config["inference_net_writer"]=inference_net_writer
    config["current_expr_start_time"]=current_time
def train_omega_actor(co_train_actor):
    # set seed
    print("test_only_omega", flush=True)
    uf.set_seed(config["seed"])
    # train the model
    eval_mse_value,eval_u_stat=co_train_actor.train_omega_neural()
    return eval_mse_value, eval_u_stat

def train_inference_actor(co_train_actor):
    #set seed
    print("train_only_inference", flush=True)
    uf.set_seed(config["seed"])
    #train the model
    EVAL_DICT=co_train_actor.train_inference_neural()
    return EVAL_DICT


def test_inference_model(co_train_actor):
    '''
    test the model
    :return: value and record
    '''
    test_epoch=1

    with torch.no_grad():
        test_dict=co_train_actor.eval_inference_model(eval_data=co_train_actor.test_loader,
                                                   eval_epoch=test_epoch,
                                                   name="test_process")
        #in the .csv
        test_df=pd.DataFrame(test_dict,index=[0])
        test_df.to_csv(config["CSV"]+"/inference_final_result.csv",mode="a",header=True)
    return test_dict
    print("tested")#

def test_omega_model(co_train_actor):
    '''
    test the model
    :return: value and record
    '''
    test_epoch=1
    with torch.no_grad():
        test_mse,u=co_train_actor.eval_omega_model(eval_data=co_train_actor.test_loader,
                                                   eval_epoch=test_epoch,
                                                   name="test_process")
        #in the .csv
        test_dict={"test_mse":test_mse.cpu().detach().numpy(),"u_stat":u.cpu().detach().numpy()}
        test_df=pd.DataFrame(test_dict,index=[0])
        test_df.to_csv(config["CSV"]+"/omega_final_result.csv",mode="a",header=True)
    print("tested")
    return test_mse,u


def expr2(expr_config,**awgs):

    """
    :param expr1_config:
    :param awgs: need_load_omgemodel:bool
                 omega_load_path:str
    :return: mse,u
    """

    # set seed
    uf.set_seed(expr_config["seed"])

    print("start train_Supervised_learning", flush=True)
    print("the prior knowledge is", expr_config["prior_knowledge"], flush=True)

    # model_init
    S_I = nn_base.Omgea_MLPwith_residual_dict(input_sample_lenth=expr_config["data_length"],
                                              hidden_dims=[512, 512, 512],
                                              output_coeff=True,
                                              hidden_act="rational",
                                              output_act="Identity",
                                              sample_vesting=2,
                                              vari_number=2
                                              )

    S_Omega = nn_base.Omgea_MLPwith_residual_dict(input_sample_lenth=expr_config["data_length"],
                                                  hidden_dims=[512, 512, 512],
                                                  output_coeff=False,
                                                  hidden_act="rational",
                                                  output_act="softmax",
                                                  sample_vesting=2,
                                                  vari_number=2
                                                  )

    # get data
    co_train_actor = co_train.train_init(S_I, S_Omega, expr_config, expr_config["train_loader"],
                                         expr_config["valid_loader"], expr_config["test_loader"],
                                         inference_net_writer, omega_net_writer)
    # dp the train
    co_train_actor.S_Omega = torch.nn.DataParallel(co_train_actor.S_Omega, device_ids=[0])
    co_train_actor.S_I = torch.nn.DataParallel(co_train_actor.S_I, device_ids=[0])
    eval_Dict=train_inference_actor(co_train_actor)
    print("we have train and valid" )
    test_inference_model(co_train_actor)
    return eval_Dict


import pandas as pd
def save_config(config):
    config_save = pd.DataFrame.from_dict(config, orient='index')
    if(os.path.exists(config["CSV"])==False):
        os.mkdir(config["CSV"])
        print("create the csv file")

    config_save.to_csv(config["CSV"]+ "/config.csv")

if __name__ == '__main__':

    #save the config to the save
    record_init(folder_num=98)
    save_config(config)
    expr2(expr_config=config)
    print("expr done")



