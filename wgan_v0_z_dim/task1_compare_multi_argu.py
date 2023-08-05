'''
this is for comparing the results of different arguments
'''
from neural_wgan_v0 import *
import json
import datetime
import utils.utils_wgan as utils_wgan#有全局变量
import utils.hyper_para as hyper_para#有全局变量
from utils.read_argu_test import *
#试验次数的全局变量
global_variable_expr = 0

def decorator_numexpr(func):
    def wrapper(*args, **kwargs):
        global global_variable_expr
        global_variable_expr += 1
        print("****record_expr_sequence***", global_variable_expr)
        return func(*args, **kwargs)
    return wrapper

def prepare(config,writer:SummaryWriter,test_num):

    #把参数保存到text
    with open(config["init_para"].save_path+'.txt', 'w') as f:
        f.write(str(config["init_para"]))
        f.close()
    # 转换成dict
    args_dict = vars(config["init_para"].arg)
    # 将字典转换为字符串
    args_string = json.dumps(args_dict)
    # 获取当前的日期和时间
    current_time = datetime.datetime.now()
    # 使用Summary.text方法上传文本数据
    writer.add_text("Text Data", args_string+str(current_time),
                    global_step=test_num)
@decorator_numexpr
def adjust_args(config):
    '''
    :param train_init_para:
    :args 原始记录参数改变一下
    '''
    const_para=config["init_para"]
    #"这里主要是参数中几乎不变的参数"
    hyper_para.args.noise_dim=config["zdimension_Gap"]
    const_para.denote = "0729_1_mse_noise_input-form"
    hyper_para.args.denote=const_para.denote
    config["init_para"].save_path = '../tb_info'+'/compare_multi_argu/args' + \
                                    const_para.denote+str(global_variable_expr)


def after_training_save_clear(config):
    #全局变量清零
    utils_wgan.count_critic_step=0
    utils_wgan.count_generator_step=0
    # 保存gif
    # images_to_video(search_path=train_init_para.save_path,
    #                 search_name="critic",
    #                 output_filename="output_critic.mov",
    #                 fps=2)
    images_to_video(search_path=config["init_para"].save_path,
                    search_name="generator",
                    output_filename="output_generator.mov",
                    fps=2)


if __name__!='__main__' :
    #后期想通过读取一个csv来做完全自动实验
    test_argus=Get_test_args()
    print(test_argus[0])
    train_init_para = train_init()
    # 使用map()函数将浮点数转换为整数

    train_init_para.zdimension_Gap=list(map(int,
                                            test_argus[0]["zdimension_Gap"]))
    train_init_para.mean=test_argus[0]["mean"]
    train_init_para.std=test_argus[0]["stddev"]
    train_init_para.seed=list(map(int,
                                  test_argus[0]["seed"]))
    train_init_para.num_epochs=test_argus[0]["epoch"]
    train_init_para.batch_size=test_argus[0]["batch_size"]
    train_init_para.beta=test_argus[0]["energy_penalty_Gap"]
    train_init_para.lr=test_argus[0]["learning_rate_range"]

