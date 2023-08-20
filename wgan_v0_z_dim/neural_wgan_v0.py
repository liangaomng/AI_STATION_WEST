
from utils.generator_discriminator import *
from utils.wgan_data import *
from utils.hyper_para import *
import ray
from ray import tune
'''
training' arguments
'''
class train_init():
    def __init__(self):
        #device
        self.the_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed=args.seed
        self.g_neural_network_width=args.g_neural_network_width
        # 训练生成器每n_generator步
        self.n_generator=args.iter_generator
        # 训练判别器每n_critice步
        self.n_critic=1
        self.beta=args.energy_penalty_argu
        # 训练GAN模型
        self.num_epochs = args.num_epochs
        self.noise_dim = args.noise_dim  # 噪声向量的维度
        self.mean = args.mean  # 高斯分布的均值
        self.stddev = args.stddev  # 高斯分布的标准差
        self.save_path=args.savepath
        self.arg=args
    def clear_attributes(self):
        self.generator=0
        self.discriminator=0
    def init_generator_discriminator(self,config):
        # 初始化生成器和判别器
        self.generator = Generator(config).to(self.the_device)
        self.discriminator = Discriminator(config).to(self.the_device)
        # 优化器-学习率
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=config["learning_rate"])
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=config["learning_rate"])
        print(self.the_device)
    def init_eval_generator(self,config):
        self.generator=Generator(config).to(self.the_device)
        self.generator.load_state_dict(torch.load(self.save_path+"/generator.pth"))
        print("load generator success")
        return self.generator


#------
'''
training
input: writer
output：csv+log
'''
def training(config,writer:SummaryWriter):

    # 初始化生成器和判别器
    config["init_para"].init_generator_discriminator(config)
    for epoch in range(config["epochs"]):
        for i,batch_data in enumerate(ode_dataloader):
            #batch_data[0]is x，batch_data[1]is y
            # 训练判别器
            config["init_para"].optimizer_d.zero_grad()
            condition_data=batch_data[0].float()#3个条件【batchsize,100,3】
            real_data_t=condition_data[:,:,1]#变量t
            real_data_t=real_data_t.reshape(-1,100)#【batch,100】
            print("real_data_t",real_data_t.shape)
            real_y_data=batch_data[1].float()#对应y的值【batchsize,100,1】

            condition_data=condition_data.reshape(-1,300)
            print("real_y_data_shape",real_y_data.shape)#【10,100】
            # 定义噪声向量的大小和分布参数
            z = torch.randn((batch_size, config["init_para"].noise_dim),device="cuda")\
                * config["stddev"]#1*noise_dim
            print("z.shape",z.shape)
            z_condition=torch.cat((z,condition_data),dim=1)#1*（noise_dim+300）
            print("z_condition",z_condition.shape)
            #这里也记录一下采样的噪声值
            #writer.add_scalar('train_critic_z', z, epoch)
            # 将噪声变量 z 写入 TensorBoard--需要研究
            #writer.add_embedding(z.numpy(), metadata=None, label_img=None, global_step=epoch)
            #生成假的y数据 要detach 不留计算图

            # fake_y_data = train_ini.generator(z_condition.detach(),
            #                                     real_data_t.detach()).float()
            #
            # fake_y_data=fake_y_data.view(batch_size,100).float()
            # #压缩成一个数
            # real_scores = torch.mean(train_ini.discriminator(real_y_data))
            # fake_scores = torch.mean(train_ini.discriminator(fake_y_data))
            # writer.add_scalars('critic_scores', {'real': real_scores.item(),
            #                                      'fake': fake_scores.item()
            #                                      }, epoch)
            # print("real_scores",real_scores)
            # # 计算判别器损失
            # d_loss = fake_scores-real_scores#神经网络打分器
            # mse=nn.MSELoss()(fake_y_data,real_y_data)
            #
            # writer.add_scalars('losses', {'mseloss':mse.item()},epoch)
            # print("dloss",d_loss)

            #plot_critic_tensor_change(train_ini,writer,d_loss,real_data_t,fake_y_data,real_y_data)

            # 反向传播和更新判别器的参数
            #d_loss.backward()
            #train_ini.optimizer_d.step()
            # #梯度截断
            # for p in train_ini.discriminator.parameters():
            #     p.data.clamp_(-args.lipschitz_clip, args.lipschitz_clip)
            #writer.add_scalars('losses', {'critic':d_loss.item()}
            #                                 , epoch)#记录
            # Log the model's weights to TensorBoard
            # for name, param in train_ini.discriminator.named_parameters():
            #     writer.add_histogram(name, param, global_step=epoch)
         # 训练生成器
        if((epoch+1)%config["init_para"].n_generator)==0:#训练生成器
                # -----------------
                #  训练生成器
                # -----------------
                config["init_para"].optimizer_g.zero_grad()
                condition_data = batch_data[0].float()  # 3个条件
                real_y_data = batch_data[1].float()  # 对应y的值
                real_data_t = condition_data[:, :, 1]  # 变量t

                condition_data=condition_data.reshape(-1,300) #[batch,300]
                data_t=real_data_t.reshape(-1,100) #[batch,100]

                supervisor_y=real_y_data#
                supervisor_y=supervisor_y.reshape(batch_size,100)

                # 生成噪声作为生成器输入
                z = torch.randn((batch_size,config["zdimension_Gap"]),device="cuda")\
                    *config["stddev"]#1*noise_dim#噪声
                z_condition=torch.cat((z,condition_data),dim=1)#（batch,noise_dim+300）
                # 生成器输出
                fake_y_data = config["init_para"].generator(z_condition,data_t)
                fake_y_data=fake_y_data.view(batch_size,100)

                print("fake_y_data shape",fake_y_data.shape)
                supervisor_y=supervisor_y.reshape(batch_size,100)
                #g_loss = -torch.mean(train_ini.discriminator(fake_y_data.float()))
                g_loss= nn.MSELoss()(fake_y_data,supervisor_y)+ torch.mean( config["init_para"].generator.energy)

                print("gloss_梯度地址",g_loss)
                plot_generator_tensor_change(config["init_para"],writer,g_loss,data_t,fake_y_data,supervisor_y,condition_data)

                writer.add_scalars('losses', {'generator':g_loss.item()}
                                             , epoch)#记录
                #记录一下 训练生成器的系数
                coeff=config["init_para"].generator.print_coeff()
                print("coeff",coeff)
                writer.add_scalars('trian_generator_coeff',
                                         {'alpha1':coeff[0,0].item(),
                                          'alpha2':coeff[0,1].item(),
                                         # 'alpha3':coeff[0,2].item(),
                                          #'alpha4':coeff[0,3].item()
                                          }
                                          , epoch)#记录

                #反向传播
                g_loss.backward(retain_graph=True)
                config["init_para"].optimizer_g.step()


        # 关闭 SummaryWriter
        writer.close()
        #保存模型
        wgan_model_save(generator=config["init_para"].generator,
                        discriminator=config["init_para"].discriminator,
                        save_path=config["init_para"].save_path)
     #每个epoch结束后打印损失
       # print(f"Epoch [{epoch+1}/{num_epochs}], Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def eval(config,writer):
    #测试模型
    #测试时用生成器和记录critic的参数量
    '''
    :param config:
    :param generator_path:
    :return: 损失
    '''
    eval_generator  = config["init_para"].init_eval_generator(config)
    eval_generator.eval()
    para_nums=count_parameters(eval_generator)
    print("参数大小",para_nums)
    writer.add_scalar("eval_generator_para_nums",para_nums,global_step=0)
    #测试生成器的损失
    eval_loss=0
    loss_avg=0
    for i, batch_data in enumerate(test_loader):  # batch_data[0]是x，batch_data[1]是y
        # 训练判别器
        condition_data = batch_data[0].float()  # 3个条件【batchsize,100,3】
        print("条件",condition_data.shape)
        real_data_t = condition_data[:, :, 1]  # 变量t
        data_t = real_data_t.reshape(-1, 100)  # [batch,100]
        condition_data = condition_data.reshape(-1, 300)
        real_y_data = batch_data[1].float()  # 对应y的 c值【batchsize,100,1】
        # 生成噪声作为生成器输入
        z = torch.randn((batch_size, config["zdimension_Gap"]), device="cuda") \
            * config["stddev"]  #噪声
        if(z.shape[1]==0):
            print("噪声为0维")
            z_condition = condition_data  # （batch,300）
        else:
            z_condition = torch.cat((z, condition_data), dim=1)  # （batch,noise_dim+300）
        # 生成器输出
        print("z_condition",z_condition.shape)
        print("data_t",data_t.shape)
        prediction_y = eval_generator(z_condition, data_t)
        prediction_y = prediction_y.view(batch_size, 100)
        eval_loss=torch.mean(nn.MSELoss()(prediction_y,real_y_data))
        loss_avg+=eval_loss

    loss_avg= loss_avg/len(test_loader)
    writer.add_scalar("eval_generator_loss",eval_loss,global_step=0)

    return loss_avg




