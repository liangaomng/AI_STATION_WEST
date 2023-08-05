'''
this is for comparing the results of different arguments
'''
from neural_wgan_v1 import *
import json
import datetime
import utils.utils_wgan as utils_wgan#有全局变量
import utils.hyper_para as hyper_para#有全局变量

def prepare(traininit,writer:SummaryWriter,test_num):

    if not os.path.exists(traininit.save_path):
        os.makedirs(traininit.save_path)
    #把参数保存到text
    with open(traininit.save_path+'.txt', 'w') as f:
        f.write(str(hyper_para.args))
        f.close()
    # 转换成dict
    args_dict = vars(traininit.arg)
    # 将字典转换为字符串
    args_string = json.dumps(args_dict)
    # 获取当前的日期和时间
    current_time = datetime.datetime.now()
    # 使用Summary.text方法上传文本数据
    writer.add_text("Text Data", args_string+str(current_time),global_step=test_num)

def adjust_args(train_init_para,test_num):
    '''
    :param train_init_para:
    :args 原始记录参数改变一下
    '''
    train_init_para.noise_dim = 100
    hyper_para.args.noise_dim=train_init_para.noise_dim
    train_init_para.denote = "mse-loss"
    hyper_para.args.denote=train_init_para.denote
    train_init_para.init_generator_discriminator(train_init_para)#初始化genertor
    train_init_para.save_path = '../tb_info/compare_multi_argu/args' + str(test_num)
    hyper_para.args.savepath=train_init_para.save_path
def after_training():
    #全局变量清零
    utils_wgan.count_critic_step=0
    utils_wgan.count_generator_step=0
    # 保存gif
    # images_to_video(search_path=train_init_para.save_path,
    #                 search_name="critic",
    #                 output_filename="output_critic.mov",
    #                 fps=2)
    images_to_video(search_path=train_init_para.save_path,
                    search_name="generator",
                    output_filename="output_generator.mov",
                    fps=2)
    # 清空属性
    train_init_para.clear_attributes()

if __name__=='__main__':
    test_list=10
    #我们衡量隐藏变量个数的超参数0-100
    #后期想通过读取一个csv来做完全自动实验
    for test_num in range(1,2):
        set_seed(42)
        # #这里args参数保存到txt“
        train_init_para = train_init()
        adjust_args(train_init_para,test_num)
        writer = SummaryWriter(train_init_para.save_path)
        prepare(train_init_para,writer,test_num)
        # #训练
        training(train_init_para,writer)
        print(f"{test_num}is done")
        after_training()
        #测试模型
        #eval()
