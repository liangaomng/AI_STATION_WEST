
import super_learn_task_expr2 as task_expr1

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
folder_num=8

def train_omega_model(config):

    print("seed",config["seed"])
    tune_id=tune.get_trial_id()
    global folder_num
    print(re.search(r'_(\d+)$', tune_id).group(1))
    folder_num=folder_num+ int(re.search(r'_(\d+)$', tune_id).group(1))
    print(folder_num)
    #update
    task_expr1.config["seed"]=config["seed"]

    task_expr1.config["Inference_num_epoch"]=config["Inference_num_epoch"]

    #train
    print("folder_num:",folder_num)
    copy_folder_path=config["src_copy_path"]
    dest_folder=task_expr1.expr_data_path_basis+f"{folder_num}_data"
    copy_folder(src_folder=copy_folder_path,dest_folder=dest_folder)
    #3 steps
    task_expr1.record_init(folder_num)
    task_expr1.save_config(task_expr1.config)
    dict=task_expr1.expr2(task_expr1.config,need_load_omgemodel=True,
                                omega_load_path=task_expr1.config["omega_model_load_path"])
    #report
    result={
              "eval_mse_4omega":0,
              "eval_u_stat_4omega":0,
    }
    tune.report(**result)
def custom_trial_str_creator(trial):
    return f"{trial.trainable_name}"+"——expr2"


if __name__ == "__main__":

    # Ray
    ray.init()

    #space
    config_space = {
        "S_Omega_lr":[1e-3,1e-4,1e-5],#constant
        "seed": tune.grid_search([i for i in range(100)]),#grid search
        "sample_method":[None,["Topk",10],["Soft_argmax",1]],#grid search
        "hidden_act":["rational"],
        "Inference_num_epoch": tune.grid_search([50000]),#grid search
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
        resources_per_trial={"cpu":8,"gpu":1}  # every trial uses one GPU
    )
  # result
    print("======================== Result =========================")
    analysis.dataframe().to_csv("/liangaoming/conda_lam/expriments/paper1/expr1/"+
                                "final_ray.csv")
    # print best config
    best_config = analysis.get_best_config(metric="eval_mse_4omega", mode="min", scope="all")
    print("Best Config: ", best_config)








