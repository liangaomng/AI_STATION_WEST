'''
this is for comparing the results of different arguments
'''
from neural_wgan_v2div import *
import json
import datetime
import utils_wgan#有全局变量

def prepare(traininit,writer:SummaryWriter,test_num):

    if not os.path.exists(traininit.save_path):
        os.makedirs(traininit.save_path)
    #把参数保存到text
    with open(traininit.save_path+'.txt', 'w') as f:
        f.write(str(args))
        f.close()
    # 转换成dict
    args_dict = vars(traininit.arg)
    # 将字典转换为字符串
    args_string = json.dumps(args_dict)
    # 获取当前的日期和时间
    current_time = datetime.datetime.now()
    # 使用Summary.text方法上传文本数据
    writer.add_text("Text Data", args_string+str(current_time),global_step=test_num)

def adjust_args(train_init_para):
    '''
    :param train_init_para:
    :args 原始记录参数改变一下
    '''
    train_init_para.noise_dim = 1 + test_num * 10
    args.noise_dim=train_init_para.noise_dim
    train_init_para.denote = "this is for the w_gan_clip to test the z dimension"
    args.denote=train_init_para.denote
    train_init_para.init_generator(train_init_para)#初始化genertor
    train_init_para.save_path = '../tb_info/compare_multi_argu/args' + str(test_num)
    args.savepath=train_init_para.save_path

if __name__=='__main__':
    test_list=3
    #后期想通过读取一个csv来做实验
    for test_num in range(0,1):
        set_seed(42)
        #这里args参数保存到txt“
        train_init_para = train_init()
        adjust_args(train_init_para)
        writer = SummaryWriter(train_init_para.save_path)
        prepare(train_init_para,writer,test_num)
        #训练
        training(train_init_para,writer)
        print(f"{test_num}is done")
        # 修改全局变量的值
        utils_wgan.count_critic_step=0
        utils_wgan.count_generator_step=0
        #清空属性
        train_init_para.clear_attributes()

