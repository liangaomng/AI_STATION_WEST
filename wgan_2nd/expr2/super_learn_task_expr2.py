#co-train

import utlis_2nd.utlis_funcs as uf
import utlis_2nd.co_train as co_train
from utlis_2nd.cusdom import *
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import warnings
import utlis_2nd.neural_base_class as nn_base
warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
from datetime import datetime
omega_net_writer = {   "train_process": 0,
                        "valid_process": 0,
                        "test_process": 0,
                        "model_checkpoint_path": 0,
                        "train_analysis_file": 0,
                        "valid_analysis_file": 0,
                        "test_analysis_file": 0}  # "train,valid,test,model_checkpoint"
inference_net_writer = {    "train_process": 0,
                            "valid_process": 0,
                            "test_process": 0,
                            "model_checkpoint_path": 0,
                            "train_analysis_file": 0,
                            "valid_analysis_file": 0,
                            "test_analysis_file": 0}  # "train,valid,test,model_checkpoint"
expr_data_path_basis="/liangaoming/conda_lam/expriments/paper1/expr1/expr1_"
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
    # NN_config
    "batch_size": 256,
    "hidden_nueral_dims": [512, 512, 512],

    "sample_vesting": 2,#unit (s)
    "hidden_act":None,
    "omega_output_act":None,
    "inference_output_act":None,
    "optimizer":None,
    "vari_number":None,

     #parameters
    "all_data_save_path": general_file_structure,
    "lamba_ini": 1,
    "lamba_grad": 1,
    "lamba_gp": 1,
    "lamba_fourier": 1,
    "prior_knowledge": {"basis_1": "x**0", "basis_2": "sin", "basis_3": "cos"},
    "S_I_lr": 1e-3,
    "S_Omega_lr": 1e-4,
    "grads_types": {"boundary": 3, "inner": 5},  # boundary:forward and backward;inner:five points
    "beta": 2,#temperature
    "freq_numbers": 51,
    # save path
    "expr_data_path": general_file_structure["train_process"],
    "CSV": general_file_structure["CSV"],
    "tb_event": general_file_structure["tb_event"],
    # writer
    "omega_net_writer": omega_net_writer,
    "inference_net_writer": inference_net_writer,
    #size of train
    "train_nomin": None,#train_size / batch_size,
    "valid_nomin": None,#valid_size / batch_size,
    "test_nomin": None,#test_size / batch_size,

    "seed": 42,
    "Omega_num_epoch": 6,
    "Inference_num_epoch": 2,
    "infer_minimum": 1e-1,

    "train_loader": None,
    "valid_loader": None,
    "test_loader": None,
    "current_expr_start_time": 0,
    "data_length":None, #t_steps
    "device":None,

}



def current2_time():
    current_time = datetime.now()
    current_start_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return current_start_time

def record_init(folder_num,expr_data_path_new):
    # record expr_time：
    current_time=current2_time()
    expr_data_path = expr_data_path_new+ f"{folder_num}"+"_data"
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

def train_omega_actor(co_train_actor,config_device):
    # set seed
    print("test_only_omega", flush=True)
    uf.set_seed(config["seed"])
    # train the model
    eval_mse_value,eval_u_stat,eval_mae=co_train_actor.train_omega_neural(device=config_device)
    return eval_mse_value, eval_u_stat,eval_mae

def train_inference_actor(co_train_actor,config_device):
    #set seed
    print("test_only_inference", flush=True)
    uf.set_seed(config["seed"])
    #train the model
    co_train_actor.train_inference_neural(device=config_device)

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


def test_omega_model(co_train_actor):
    '''
    test the model
    :return: value and record
    '''
    test_epoch=1

    with torch.no_grad():
        test_mse,u,test_mae=co_train_actor.eval_omega_model(eval_data=co_train_actor.test_loader,
                                                   eval_epoch=test_epoch,
                                                   name="test_process")
        #in the .csv
        test_dict={"test_mse":test_mse.cpu().detach().numpy(),
                   "u_stat":u.cpu().detach().numpy(),
                   "test_mae":test_mae.cpu().detach().numpy()}
        test_df=pd.DataFrame(test_dict,index=[0])
        test_df.to_csv(config["CSV"]+"/omega_final_result.csv",mode="a",header=True)
    print("tested omega done")
    return test_mse,u,test_mae


def expr(expr_config,train_type="omega_net"):

    # set seed
    uf.set_seed(expr_config["seed"])

    print("the prior knowledge is", expr_config["prior_knowledge"], flush=True)


    #model_init
    S_I = nn_base.Omgea_MLPwith_residual_dict(     input_sample_lenth=expr_config["data_length"],
                                                   hidden_dims=expr_config["hidden_nueral_dims"],
                                                   output_coeff=True,
                                                   hidden_act=expr_config["hidden_act"] , #"rational"
                                                   output_act=expr_config["inference_output_act"],#"identity"
                                                   sample_vesting=expr_config["sample_vesting"],
                                                   vari_number=expr_config["vari_number"],
                                                   device_type=expr_config["device"]
                                             )


    S_Omega =nn_base.Omgea_MLPwith_residual_dict(  input_sample_lenth=expr_config["data_length"],
                                                   hidden_dims=expr_config["hidden_nueral_dims"],
                                                   output_coeff=False,
                                                   hidden_act=expr_config["hidden_act"], # "rational"
                                                   output_act=expr_config["omega_output_act"],#softmax
                                                   sample_vesting=expr_config["sample_vesting"],
                                                   vari_number=expr_config["vari_number"],
                                                   device_type=expr_config["device"]
                                                  )

    # get data
    co_train_actor = co_train.train_init(S_I, S_Omega, expr_config, expr_config["train_loader"],
                                         expr_config["valid_loader"], expr_config["test_loader"],
                                         inference_net_writer,omega_net_writer)
    # dp the train-cuda
    if(expr_config["device"]=="cuda"):
        co_train_actor.S_Omega = torch.nn.DataParallel(co_train_actor.S_Omega, device_ids=[0])
        co_train_actor.S_I = torch.nn.DataParallel(co_train_actor.S_I, device_ids=[0])
    else:
        co_train_actor.S_Omega.to(device=expr_config["device"])
        co_train_actor.S_I.to(device=expr_config["device"])

    if (train_type=="omega_net"):
        train_omega_actor(co_train_actor,config_device=expr_config["device"])
        mse_value,u_stat,eval_mae=test_omega_model(co_train_actor)

        return  mse_value,\
        u_stat,\
        eval_mae,\
        co_train_actor

    elif (train_type=="inference_net"):
        print("hi")
        train_inference_actor(co_train_actor,config_device=expr_config["device"])
        test_dict=test_inference_model(co_train_actor)

        return test_dict,\
               co_train_actor

    print("we have train and valid" )




import pandas as pd
def save_config(config):
    config_save = pd.DataFrame.from_dict(config, orient='index')
    if(os.path.exists(config["CSV"])==False):
        os.mkdir(config["CSV"])
        print("create the csv file")

    config_save.to_csv(config["CSV"]+ "/config.csv")


def do_expr(results_save_path=None,
          folder_num=None,
          train_config=None,
          model_type=None):
    '''
    :param
        1.path:str
        2.folder_num: the folder number-int
        3.train_config: the config of the train -dict
        4.model_type=None: the type of the model omega_net or inference_net

    :return: results about mse,u,mae,
    '''
    #save the config to the save

    record_init(folder_num=folder_num,expr_data_path_new=results_save_path)
    save_config(train_config)
    return expr(expr_config=train_config,train_type=model_type)




if __name__ == '__main__':
    do_expr()




