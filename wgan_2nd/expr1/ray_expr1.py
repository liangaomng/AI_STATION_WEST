
import super_learn_task_expr1 as task_expr1

import ray
from ray import tune
from ray.tune import CLIReporter
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示ERROR日志

def copy_folder(src_folder,dest_folder):
    '''

    :param src_folder:  son’ flies
    :param dest_folder: father
    :return:
    '''
    # dest_folder: the folder you want to paste
    # check
    #son to father
    if not os.path.exists(src_folder):
        print(f"not exist{src_folder}.")
        return
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print("dest_folder not exist,create it.")

    # copy
    try:
        for item in os.listdir(src_folder):
            src_path = os.path.join(src_folder, item)
            dest_path = os.path.join(dest_folder, item)

            # file
            if os.path.isfile(src_path):
                shutil.copy(src_path, dest_path)

            # folder
            elif os.path.isdir(src_path):
                shutil.copytree(src_path, dest_path)
    except Exception as e:
        print(f"erro: {e}")

import re
folder_num=98

def train_omega_model(config):

    print("seed",config["seed"])
    tune_id=tune.get_trial_id()
    global folder_num
    print(re.search(r'_(\d+)$', tune_id).group(1))
    folder_num=folder_num+ int(re.search(r'_(\d+)$', tune_id).group(1))
    print(folder_num)
    #update
    task_expr1.config["seed"]=config["seed"]
    task_expr1.config["S_Omega_lr"]=config["S_Omega_lr"]
    task_expr1.config["Omega_num_epoch"]=config["omega_epochs"]
    task_expr1.config["freq_num"]=config["freq_numbers"]
    task_expr1.config["beta"]=config["beta"]
    task_expr1.config["Omega_num_epoch"]=config["Omega_num_epoch"]
    #train
    print("folder_num:",folder_num)
    copy_folder_path=config["src_copy_path"]
    dest_folder=task_expr1.expr_data_path_basis+f"{folder_num}_data"
    copy_folder(src_folder=copy_folder_path,dest_folder=dest_folder)
    #3 steps
    task_expr1.record_init(folder_num)
    task_expr1.save_config(task_expr1.config)
    eval_mse_4omega,eval_u_stat_4omega,eval_mae=task_expr1.expr1(task_expr1.config)
    #report
    result={
              "eval_mse_4omega":eval_mse_4omega.cpu().detach().numpy(),
              "eval_u_stat_4omega":eval_u_stat_4omega.cpu().detach().numpy(),
               "eval_mae":eval_mae.cpu().detach().numpy(),
    }
    tune.report(**result)
def custom_trial_str_creator(trial):
    return f"{trial.trainable_name}"+"——expr1"


if __name__ == "__main__":

    # Ray
    ray.init()

    #space
    config_space = {
        "S_Omega_lr": task_expr1.config["S_Omega_lr"],#constant
        "omega_epochs":task_expr1.config["Omega_num_epoch"], #constnt
        "seed": tune.grid_search([1]),#grid search
        "freq_numbers": tune.grid_search([1]),#grid search
        "beta": tune.grid_search([1]),#grid search
        "Omega_num_epoch": tune.grid_search([2000]),#grid search
        "src_copy_path":"/liangaoming/conda_lam/expriments/paper1/expr_1",

    }
   # reporter
    reporter = CLIReporter(metric_columns=["eval_mse_4omega", "u_stat_4omega"])
    # run
    analysis = tune.run(
        train_omega_model,
        config=config_space,
        metric="eval_mse_4omega",
        mode="min",
        trial_name_creator=custom_trial_str_creator,
        progress_reporter=reporter,
        local_dir="/liangaoming/conda_lam/expriments/paper1/expr1/tb_info/ray_expr1",
        resources_per_trial={"cpu":4,"gpu":1}  # every trial uses one GPU
    )
  # result
    print("======================== Result =========================")
    analysis.dataframe().to_csv("/liangaoming/conda_lam/expriments/paper1/expr1/"+
                                "final_ray.csv")
    # print best config
    best_config = analysis.get_best_config(metric="eval_mse_4omega", mode="min", scope="all")
    print("Best Config: ", best_config)








