import torch
import torch.nn as nn
import sympy as sp
import numpy as np


import matplotlib.pyplot as plt
'''
    class: generator
    define generator network 
    input: n dimension noise 
    output: a solution which combine different basis
'''

function_symbol=[]


def clear_str_list():

    global function_symbol
    function_symbol=[]
# 自定义层
class Basis_Transform(nn.Module):
    def __init__(self, input_dim, output_dim,basis_num=2,omega=0):

        super(Basis_Transform, self).__init__()

        self.x = sp.symbols('x')
        # create func matrix -prior
        self.func_matrix = sp.Matrix([self.x ** i for i in range(basis_num)])

        #self.func_matrix = sp.Matrix([self.x ** i for i in range(basis_num)])
        #
        # self.x = sp.symbols('x')
        # # create func matrix -prior
        # #self.func_matrix = sp.Matrix([self.x ** i for i in range(basis_num)])
        # #for test
        # self.omega=nn.Parameter(torch.tensor(0,dtype=torch.float64))
        # self.func_matrix= sp.Matrix([0,sp.sin(self.omega*self.x),sp.cos(self.omega*self.x)])
        # #convert to function input x, output is the value of the function
        # self.funcs = [sp.lambdify(self.x, self.func_matrix[i]) for i in range(basis_num)]
        # # f_symbolic is symbols
        # self.f_symbolic = [self.func_matrix[i] for i in range(basis_num)]
        # # result_basis is the result of the basis function [100,basis_num]
        # self.result_basis = torch.zeros(100, basis_num, dtype=torch.float64, device='cuda')
        # self.result_basis = self.A_matrix()
        # self.result_basis = nn.Parameter(self.result_basis, requires_grad=False)
        # #coeffs
        # self.coeffs=nn.Parameter(torch.tensor(0,dtype=torch.float64))
        # self.coeffs.requires_grad_(False)
        # #energy
        # self.energy=nn.Parameter(torch.tensor(0,dtype=torch.float64))
        # self.energy.requires_grad_(False)
        # self.basis_num=basis_num


    def updata_basis(self,omega):

        self.omega=omega
        batch_size, _ = self.omega.size()

        self.func_matrix = sp.Matrix([0, sp.sin(self.omega * self.x), sp.cos(self.omega * self.x)])
        # convert to function input x, output is the value of the function
        self.funcs = [sp.lambdify(self.x, self.func_matrix[i]) for i in range(self.basis_num)]
        # f_symbolic is symbols
        self.f_symbolic = [self.func_matrix[i] for i in range(self.basis_num)]
        # result_basis is the result of the basis function [100,basis_num]
        self.result_basis = torch.zeros(100, self.basis_num,
                                        dtype=torch.float64, device='cuda')
        self.result_basis = self.A_matrix()
        #analy_gradients
        #add the gradient
        # [batch,100,1]
        global function_symbol
        function_symbol.append(self.f_symbolic)


    def A_matrix(self):

        data_t_numpy=np.linspace(0,2,100)
        # return the A matrix
        for i, func in enumerate(self.funcs):
            value=func(data_t_numpy)
            value=torch.tensor(value,dtype=torch.float64)
            self.result_basis[:, i] = value

        return self.result_basis
    def stat(self):
        '''
        :param coeffs: [batch,basis_num*2]
        :return: energy[batch,2]
        '''

        batch_size, _ = self.coeffs.size()
        self.energy = torch.zeros(batch_size,2, dtype=torch.float64, device='cuda')

        for i in range(batch_size):
            for j in range(self.basis_num):
                #z1 energy
                self.energy[i,0] += (self.coeffs[i, j] ** 2) * torch.norm(self.result_basis[:, j], p=2)
                #z2 energy
                self.energy[i,1]+= (self.coeffs[i, j+self.basis_num] ** 2) * torch.norm(self.result_basis[:, j], p=2)

        return self.energy

    def forward(self,x,omega):

        self.updata_basis(omega)

        # input x:[batch,basis_num*2] coeffs
        batch_size, _= x.size()
        self.coeffs = x#[batch,basis_num*2]

        # x is the coeffs
        x = x.view(batch_size,self.basis_num ,-1)  # [batch,basis_num,2]
        #cal energy
        self.stat()

        fake=torch.zeros(batch_size,100,2,device="cuda")

        # matmul with the weight
        for i in range(batch_size):
            #z1_t
            fake[i,:,0] = torch.matmul(self.result_basis,
                                       x[i,:,0])
            #z2_t
            fake[i,:,1] = torch.matmul(self.result_basis,
                                       x[i,:,1])
        #fake is output [batch,100,2]

        return fake,self.energy,x,self.result_basis

class omega_generator(nn.Module):
    def __init__(self,input_dim=202,output_dim=1):
        super(omega_generator, self).__init__()
        self.omega_fc1=nn.Sequential(
            nn.Linear(input_dim,254),
            nn.LeakyReLU(0.2),
            nn.Linear(254,128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,output_dim),
        )
        # kaiming_init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')

    def forward(self,x):
        '''
        :param x: [batch,101,2] ini &gradinets 101*2
        :return: [batch,1]
        '''
        x=x.view(-1,202)
        self.omega=self.omega_fc1(x)
        #discrete
        self.omega=RoundWithPrecisionSTE.apply(self.omega,2)

        return self.omega
#there is a matrix

class Generator(nn.Module):
    '''
    there are four items in the class
    '''
    '''
    init fuction
    '''
    def __init__(self,config:dict,basis_num,omega_value):

        super(Generator, self).__init__()
        hidden_width=config["g_neural_network_width"]
        #structure
        self.model_coeff = nn.Sequential(
            nn.Linear(config["zdimension_Gap"]+10, hidden_width),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_width, hidden_width),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_width, basis_num*2), #2 is the z1 and z2
        )
        #condition
        self.cond_fc1=nn.Sequential(
            nn.Linear(202,254),
            nn.LeakyReLU(0.2),
            nn.Linear(254,128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,10),
        )
        # kaiming_init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        self.fake_data=torch.zeros(config["batch_size"],100,2,device="cuda").requires_grad_(True)

    '''
    func: forward
    input: noise + data_t
    return :data,energy,coeffs
    '''
    def return_symbol(self):
        return self.f_symbolic

    def put_in_the_matrix(self):
        '''
        :param coeffs: [batch,basis_num,2]
        :return: fake [batch,100,2]，
        :note: 100 is the time step
        '''
        _, coeff_number, z_number = self.coeff.size()
        # matmul with the weight and generate the fake data
        # [100,1]=[100,basis_num]*[basis_num,1]
        # z1_t [batch,100,1]
        self.fake_data[:, :, 0] = torch.matmul(self.result_basis,
                                                   self.coeff[:, :,0])
        # z2_t [batch,100,1]
        self.fake_data[:, :, 1] = torch.matmul(self.result_basis,
                                                   self.coeff[:, :,1])
        # fake is output [batch,100,2]
        return self.fake_data

    def forward(self,x,condition,omega_value,basis_num=3):
        #condition is [batch,101,2]
        #cat the noise and data_t
        condition=condition.view(-1,202)
        #embed the condition to 10 dim
        condition_out=self.cond_fc1(condition)
        # x is [batch,noise+10]

        x=torch.cat((x,condition_out),dim=1)
        #cal the coeff 【batch，basis_num,2】
        self.coeff=self.model_coeff(x)
        self.coeff=self.coeff.reshape(-1,basis_num,2)
        # self.coeff=RoundWithPrecisionSTE.apply(self.coeff,1)
        fake=self.put_in_the_matrix()
        print(fake)
        exit()
        #update symbolic now_ we


        return fake,energy,coeffs,basis_matrix


class RoundWithPrecisionSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, decimal_places=0):
        multiplier = 10 ** decimal_places
        return torch.round(input * multiplier) / multiplier

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # identity gradient


'''
class: discriminator
critic network input [z1_t,z2_t] [100,2]
now there is no condition in it
output:score scalar
'''
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # flatten[100,2]
        self.flatten = nn.Flatten()

        self.model = nn.Sequential(
            nn.Linear(400, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),

        )
        #kaiming init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight,a=0,mode='fan_in')

    def forward(self, x,grad):
        #grad is [batch,100,2]
        grad=self.flatten(x)
        x=self.flatten(x)
        x=torch.cat((x,grad),dim=1)
        critic=self.model(x)
        return critic

def wgan_model_save(generator:nn.Module,
                    discriminator:nn.Module,
                    save_path:str):

    torch.save(generator.state_dict(), save_path+'/generator.pth')
    torch.save(discriminator.state_dict(), save_path+'/discriminator.pth')


'''
func:computer w_div
input: real_samples,real_out,fake_samples,fake_out
output:scalr -div
'''
def compute_w_div(real_samples, real_out, fake_samples, fake_out):
    # define parameters
    k = 2
    p = 6
    # cal the gradient of real samples
    weight = torch.full((real_samples.size(0),1), 1, device='cuda')

    #adjust the weight
    real_grad = torch.autograd.grad(outputs=real_out,
                              inputs=real_samples,
                              grad_outputs=weight,
                              create_graph=True,
                              retain_graph=True, only_inputs=True)[0]

    # L2 real norm
    real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1)

    # fake gradient
    fake_grad = torch.autograd.grad(outputs=fake_out,
                              inputs=fake_samples,
                              grad_outputs=torch.ones_like(fake_out),
                              create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
    # L2 fake norm
    fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1)
    # div_gp
    div_gp = torch.mean(real_grad_norm ** (p / 2) + fake_grad_norm ** (p / 2)) * k / 2
    return div_gp
