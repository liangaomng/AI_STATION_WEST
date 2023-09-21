#
import ray
from ray import tune
from ray.tune import CLIReporter
import super_learn_task_expr1 as super_learn_expr1
import torch
def train_model(config):

    global trial_time
    trial_time=trial_time+1
    super_learn_expr1.config["save_plot_tb_path"]=".../tb_info/super_expr"+f"{trial_time}"+"_env"
    #record
    current_device_id = torch.cuda.current_device()
    print(f"Running on GPU_main: {current_device_id}")
    eval_mse_4omega,u_stat_4omega=super_learn_expr1.expr1()

    result={
        "eval_mse_4omega":eval_mse_4omega,
        "u_stat_4omega":u_stat_4omega,
    }

    tune.report(**result)


trial_time=1

import json
if __name__ == "__main__":

    # Ray
    ray.init(local_mode=True)  # local
    #space
    config_space = {
        "S_Omega_lr": super_learn_expr1.config["learning_rate"],#constant
        "omega_epochs":  super_learn_expr1.config["Omega_num_epoch"], #constnt
        "seed": tune.grid_search([1,42,100]),#grid search
        "freq_numbers": tune.grid_search([1,2,3,4]),#grid search
        "beta": tune.grid_search([0.1,1,10,100]),#grid search
    }
  # 配置Tune的报告器和调度器
    reporter = CLIReporter(metric_columns=["eval_mse_4omega", "u_stat_4omega"])

    # 使用Ray Tune来搜索最佳的超参数组合，并将结果记录到TensorBoard中
    analysis = tune.run(
        train_model,
        config=config_space,
        metric="eval_mse_4omega",
        mode="min",
        progress_reporter=reporter,
        local_dir="../tb_info/ray_expr1",  # TensorBoard日志目录
        resources_per_trial={"gpu": 1}  # every trial uses one GPU
    )
  # 得到最后的结果
    print("======================== Result =========================")
    print(analysis.dataframe().to_csv("final.csv"))
    # 获取最优的参数配置
    best_config = analysis.get_best_config(metric="eval_mse_4omega", mode="min", scope="all")
    with open("best_config.json", "w") as file:
        json.dump(best_config, file, indent=4)

    print("Best Config saved to best_config.json")






