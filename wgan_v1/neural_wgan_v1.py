
from utils.generator_discriminator import *
from utils.wgan_data import *
from utils.hyper_para import *
'''
training' arguments
'''
class train_init():
    def __init__(self):
        #device
        self.the_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.generator = Generator(config).to(self.the_device)
        self.discriminator = Discriminator(config).to(self.the_device)
        # 优化器-学习率
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=config["learning_rate"])
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=config["learning_rate"])
        print(self.the_device)


    #------
'''
training
input: writer
output：csv+log
'''
def training(train_ini:train_init,writer:SummaryWriter):

    for epoch in range(train_ini.num_epochs):
        for i,batch_data in enumerate(ode_dataloader):#batch_data[0]是x，batch_data[1]是y
            # 训练判别器
            train_ini.optimizer_d.zero_grad()
            condition_data=batch_data[0].float()#3个条件【batchsize,100,3】
            real_data_t=condition_data[:,:,1]#变量t
            real_data_t=real_data_t.reshape(-1,100)#【batch,100】
            print("real_data_t",real_data_t.shape)
            real_y_data=batch_data[1].float()#对应y的值【batchsize,100,1】

            condition_data=condition_data.reshape(-1,300)
            print("real_y_data_shape",real_y_data.shape)#【10,100】
            # 定义噪声向量的大小和分布参数
            z = torch.randn((batch_size,train_ini.noise_dim),device="cuda")\
                *train_ini.stddev#1*noise_dim
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
        if((epoch+1)%train_ini.n_generator)==0:#训练生成器
                # -----------------
                #  训练生成器
                # -----------------
                train_ini.optimizer_g.zero_grad()
                condition_data = batch_data[0].float()  # 3个条件
                real_y_data = batch_data[1].float()  # 对应y的值
                real_data_t = condition_data[:, :, 1]  # 变量t

                condition_data=condition_data.reshape(-1,300) #[batch,300]
                data_t=real_data_t.reshape(-1,100) #[batch,100]

                supervisor_y=real_y_data#
                supervisor_y=supervisor_y.reshape(batch_size,100)

                # 生成噪声作为生成器输入
                z = torch.randn((batch_size,train_ini.noise_dim),device="cuda")\
                    *train_ini.stddev#1*noise_dim#噪声
                z_condition=torch.cat((z,condition_data),dim=1)#（batch,noise_dim+300）
                # 生成器输出
                fake_y_data = train_ini.generator(z_condition,data_t)
                fake_y_data=fake_y_data.view(batch_size,100)

                print("fake_y_data shape",fake_y_data.shape)
                supervisor_y=supervisor_y.reshape(batch_size,100)
                #g_loss = -torch.mean(train_ini.discriminator(fake_y_data.float()))
                g_loss= nn.MSELoss()(fake_y_data,supervisor_y)+ torch.mean(train_ini.generator.energy)

                print("gloss_梯度地址",g_loss)
                plot_generator_tensor_change(train_ini,writer,g_loss,data_t,fake_y_data,supervisor_y)

                writer.add_scalars('losses', {'generator':g_loss.item()}
                                             , epoch)#记录
                #记录一下 训练生成器的系数
                coeff=train_ini.generator.print_coeff()
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
                train_ini.optimizer_g.step()


        # 关闭 SummaryWriter
        writer.close()
        #保存模型
        wgan_model_save(generator=train_ini.generator,
                        discriminator=train_ini.discriminator,
                        save_path=train_ini.save_path)
     #每个epoch结束后打印损失
       # print(f"Epoch [{epoch+1}/{num_epochs}], Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def eval(train_ini,generator_path,discriminatpr_path):
    #测试模型
    #加载模型
    eval_generator  = train_ini.generator()
    eval_generator.load_state_dict(torch.load(generator_path))
    eval_discriminator = train_ini.generator()
    eval_discriminator.load_state_dict(torch.load(discriminatpr_path))

    eval_generator.eval()
    eval_discriminator.eval()
    with torch.no_grad():
        prediction_y=eval_generator()
        #计算损失
        eval_loss=nn.MSELoss()(prediction_y,real_y)




