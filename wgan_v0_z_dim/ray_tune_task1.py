#
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from task1_compare_multi_argu import *
def train_model(config):
    #调节记录的参数
    global global_variable_expr
    adjust_args(config)
    writer = SummaryWriter(config["init_para"].save_path)
    print("地址",config["init_para"].save_path)
    prepare(config,writer=writer,
            test_num=global_variable_expr)
    #训练
    training(config, writer)
    after_training_save_clear(config)
    loss=eval(config,writer)

    tune.report(loss=loss.item(),record_loss=loss,l=1)




if __name__ == "__main__":
    print(os.getenv("PYTHONPATH"))
    print(os.getcwd())
    print(train_init_para.zdimension_Gap)
    # 初始化Ray
    ray.init(local_mode=True)  # 在本地模式下运行
    # 定义超参数搜索空间
    config_space = {
        "learning_rate": train_init_para.lr,#离散搜索
        "batch_size": train_init_para.batch_size, #常数
        "epochs": train_init_para.num_epochs, #常数
        "zdimension_Gap": tune.grid_search(train_init_para.zdimension_Gap),#网格搜索
        "mean": train_init_para.mean,
        "stddev": tune.choice(train_init_para.std),
        "seed": tune.grid_search(train_init_para.seed),#网格搜索
        "energy_penalty_Gap": tune.choice(train_init_para.beta),
        "g_neural_network_width": train_init_para.g_neural_network_width,
        "init_para":train_init_para,
    }
  # 配置Tune的报告器和调度器
    reporter = CLIReporter(metric_columns=["loss"])

    # 使用Ray Tune来搜索最佳的超参数组合，并将结果记录到TensorBoard中
    analysis = tune.run(
        train_model,
        config=config_space,
        metric="loss",
        mode="min",
        num_samples=10,  # 要尝试的超参数组合数量
        progress_reporter=reporter,
        local_dir="../tb_info/ray_tune",  # TensorBoard日志目录
    )
  # 得到最后的结果
    print("======================== Result =========================")
    print(analysis.dataframe().to_csv("final.csv"))
    # 获取最优的参数配置
    best_config = analysis.get_best_config(metric="loss", mode="min", scope="all")
    print("Best Config:")
    print(best_config)





