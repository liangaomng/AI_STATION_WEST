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
class Generator(nn.Module):
    '''
    there are four items in the class
    '''
    '''
    init fuction
    '''
    def __init__(self,config:dict):

        super(Generator, self).__init__()
        hidden_width=config["g_neural_network_width"]
        #structure
        self.model = nn.Sequential(
            nn.Linear(config["zdimension_Gap"], hidden_width),
            nn.LeakyReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.LeakyReLU(),
            nn.Linear(hidden_width, 4*2),
            nn.LeakyReLU()
        )
        # kaiming_init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        self.batch_size= config["batch_size"]
        self.energy_regurla = 0  # regulariztion
        self.energy = torch.zeros(self.batch_size, 1, device="cuda")  # dot with itself
        self.generate_data = torch.zeros(self.batch_size,100,2, device="cuda")  # generate data z1_t and z2_t
        self.math_matrix = []  # basis function
        print("***generator_init")
    def genertor_policy(self):
        '''
        according the frequency of the basis function to generate the policy
        :return:
        '''
        pass

    '''
    func: create_matrix
    meaning:accoring to the basis function to return column of
    each basis data
    retun different function basis data of data_t and list of str of basis
    '''
    def create_basis_matrix(self,data_t): #basis function

        basis_num=4
        data_t_numpy=data_t.cpu().clone().numpy()
        #create symbol
        x = sp.symbols('x')
        # create func matrix -prior
        self.func_matrix = sp.Matrix([x**i for i in range(basis_num)])
        print("math_matrix",self.func_matrix)

        #convert to function. input x, output is the value of the function
        funcs = [sp.lambdify(x, self.func_matrix[i]) for i in range(basis_num)]

        #create result matrix
        result_basis=np.zeros((self.batch_size,100,basis_num))
        for i,func in enumerate(funcs):
            result_basis[:,:,i]=func(data_t_numpy[:,:,0])

        # f_symbolic is symbols
        self.f_symbolic =[self.func_matrix[i] for i in range(4)]
        print("f_symbolic",self.f_symbolic) #like [1,x,x**2,x**3]

        #return differnt the basis function f(data_t) [batch,100,4]
        #and the str symbol of the basis function [1,x,x**2,x**3]
        return result_basis,self.f_symbolic


    def print_coeff(self):
        print('coeff is',self.coeff)
        return self.coeff

    #calculate the data energy and coeffs,
    #suppose 4 basis function
    #input:    coeff[batch,4,2] z1_t and z2_t
    #output     generator data[batch,100,2] z1_t and z2_t
    def calculate_generate(self,coeff,data_t):

        #4 prior --future to control the prior
        basis_matrix=np.ones((self.batch_size,100,4))
        basis_matrix,basis_str=self.create_basis_matrix(data_t)

        #convert to tensor
        num_matirx_tensor=torch.from_numpy(basis_matrix).to(device="cuda")

        #2 column data is  [batch,100,2]---generate_data: z1_t & z2_t
        for i in range(self.batch_size):
            a=coeff[i,0]*num_matirx_tensor[i,:,0] \
                      +coeff[i,1]*num_matirx_tensor[i,:,1]\
                      +coeff[i,2]*num_matirx_tensor[i,:,2]\
                       +coeff[i,3]*num_matirx_tensor[i,:,3]
            b=coeff[i,4]*num_matirx_tensor[i,:,0] \
                        +coeff[i,5]*num_matirx_tensor[i,:,1]\
                        +coeff[i,6]*num_matirx_tensor[i,:,2]\
                        +coeff[i,7]*num_matirx_tensor[i,:,3]

            #z1_t
            self.generate_data[i,:,0]=a
            #z2_t
            self.generate_data[i, :, 1] = b
            # cal the energy of the sum basis
            self.energy[i,:]=(coeff[i,0]**2)*torch.norm(num_matirx_tensor[i,:,0],p=2)\
               +(coeff[i,1]**2)*torch.norm(num_matirx_tensor[i,:,1],p=2)\
               +(coeff[i,2]**2)*torch.norm(num_matirx_tensor[i,:,2],p=2)\
               +(coeff[i,3]**2)*torch.norm(num_matirx_tensor[i,:,3],p=2)\
               +(coeff[i,4]**2)*torch.norm(num_matirx_tensor[i,:,0],p=2)\
               +(coeff[i,5]**2)*torch.norm(num_matirx_tensor[i,:,1],p=2)\
               +(coeff[i,6]**2)*torch.norm(num_matirx_tensor[i,:,2],p=2)\
               +(coeff[i,7]**2)*torch.norm(num_matirx_tensor[i,:,3],p=2)

        print("generate_data_shape", self.generate_data.shape)

        return self.generate_data,basis_matrix,basis_str

    '''
    func: forward
    input: noise + data_t
    return :data,energy,coeffs
    '''
    def forward(self,x,data_t):
        #check the input data_t=[100*1]
        assert data_t.shape[1]==100 and data_t.shape[2]==1
        self.coeff=self.model(x)
        data,basis_matrix,basis_str=self.calculate_generate(self.coeff,data_t)
        return data,self.energy,self.coeff,basis_matrix,basis_str


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
            nn.Linear(200, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )
        #kaiming init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight,a=0,mode='fan_in')

    def forward(self, x):
        print(x.shape)
        x=self.flatten(x)
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
    print("weight",weight.shape)
    print("real_out",real_out.shape)
    print("real_samples",real_samples.shape)
    #adjust the weight
    real_grad = torch.autograd.grad(outputs=real_out,
                              inputs=real_samples,
                              grad_outputs=weight,
                              create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
    # L2 real norm
    real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1)
    print(real_grad_norm.shape)
    # fake gradient
    fake_grad = torch.autograd.grad(outputs=fake_out,
                              inputs=fake_samples,
                              grad_outputs=weight,
                              create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
    # L2 fake norm
    fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1)
    # div_gp
    div_gp = torch.mean(real_grad_norm ** (p / 2) + fake_grad_norm ** (p / 2)) * k / 2
    return div_gp
