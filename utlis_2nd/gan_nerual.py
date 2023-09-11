import torch
import torch.nn as nn
import sympy as sp
import numpy as np
from sympy import Symbol
import utlis_2nd.utlis_funcs as uf
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

class omega_generator(nn.Module):
    def __init__(self,input_dim=202,output_dim=2):
        super(omega_generator, self).__init__()
        self.rational_function_1=Rational()
        self.rational_function_2=Rational()

        self.omega_fc1=nn.Sequential(
            nn.Linear(input_dim,254),
            self.rational_function_1,
            nn.Linear(254,128),
            self.rational_function_2,
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
        #self.omega=RoundWithPrecisionSTE.apply(self.omega,1)

        return self.omega


class Rational(torch.nn.Module):
    """Rational Activation function.
    Implementation provided by Mario Casado (https://github.com/Lezcano)
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are initialized to the best rational
    approximation of degree (3,2) to the ReLU function
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """
    def __init__(self):
        super().__init__()
        self.coeffs = torch.nn.Parameter(torch.Tensor(4, 2))
        self.reset_parameters()

    def reset_parameters(self):
        self.coeffs.data = torch.Tensor([[1.1915, 0.0],
                                         [1.5957, 2.383],
                                         [0.5, 0.0],
                                         [0.0218, 1.0]])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.coeffs.data[0,1].zero_()
        exp = torch.tensor([3., 2., 1., 0.], device=input.device, dtype=input.dtype)
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        output = torch.div(PQ[..., 0], PQ[..., 1])
        return output


class Generator(nn.Module):
    '''
    Generator has two neural networks
    '''
    def __init__(self,config:dict,basis_num,omega_value):

        super(Generator, self).__init__()
        hidden_width=config["g_neural_network_width"]
        self.rational_function_1=Rational()
        self.rational_function_2=Rational()
        #structure of model_coeff
        self.model_coeff = nn.Sequential(
            nn.Linear(config["zdimension_Gap"]+10, hidden_width),
            self.rational_function_1,
            nn.Linear(hidden_width, hidden_width),
            self.rational_function_2,
            nn.Linear(hidden_width, basis_num*2), #2 is the z1 and z2

        )
        self.rational_function_3 = Rational()
        self.rational_function_4 = Rational()
        #condition of embedding
        self.cond_fc1=nn.Sequential(
            nn.Linear(202,254),
            self.rational_function_3,
            nn.Linear(254,128),
            self.rational_function_4,
            nn.Linear(128,10),

        )
        # kaiming_init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        #self.fake_data=torch.zeros(config["batch_size"],100,2,device="cuda").requires_grad_(True)

    def forward(self,x,condition,omega_value):
        '''
        input:      x is noise [batch,noise_dimension],
                    condition is[batch,101,2]
                    omega_value is [batch,2]
        output:     fake_coeffs:[batch,basis_num*2]
        '''
        #condition is [batch,101,2]
        condition=condition.view(-1,202)

        #embedd the condition to 10 dim for condition_out
        condition_out=self.cond_fc1(condition)
        # x is [batch,noise+10]
        x=torch.cat((x,condition_out),dim=1)

        #fake coeff is [batch,num_basis*2]
        fake_coeff=self.model_coeff(x)

        return fake_coeff


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
    def __init__(self,input_dim=2*100+101*2,output=1):
        super(Discriminator, self).__init__()

        # flatten
        self.flatten = nn.Flatten()
        self.rational_function = Rational()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            self.rational_function,
            nn.Linear(512, 64),
            self.rational_function,
            nn.Linear(64, 1),
            self.rational_function

        )
        #kaiming init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight,a=0,mode='fan_in')

    def forward(self, x,condition_info):
        #condition_info is [batch,100,2]
        real=self.flatten(x)

        condition_info=self.flatten(condition_info)
        # after cat is [batch,100*2+101*2]
        x=torch.cat((real,condition_info),dim=1)
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
    '''
    func:computer w_div
    input: real_samples,real_out,fake_samples,fake_out
    output:scalar -div
    '''
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

def to_symbol(dict):
    '''
    :param dict:
    :return: [], conclude the symbol
    '''
    symbols = []
    x = sp.symbols('x')
    for key, value in dict.items():
        if isinstance(value, str):
            if value == 'sin':
                symbols.append(sp.sin(x))
            elif value == 'cos':
                symbols.append(sp.cos(x))
            elif value =='x**0':
                symbols.append(x**0)
        else:
            symbols.append(value)
    return symbols

#func:pick_basis_function
#input:dict
#output:basis_function
def pick_basis_function(basis_dict:dict,basis_num=3):
    '''
    :param 1. basis_dict:is a dict like  {'basis_1': 0, 'basis_2': 'sin', 'basis_3': 'cos'}
           2. numbers=3 means that we pick the 3 basis functions
    :return: #Matrix([[0], [sin(x)], [cos(x)]])
    '''
    # create func matrix -prior
    symbols=to_symbol(basis_dict)#[0, sin(x), cos(x)]
    #covert to the func_matrix
    func_matrix = sp.Matrix([symbols[i] for i in range(basis_num)])
    return func_matrix

def Get_basis_function_info(dict,numbers=3):
    '''
    :param:     1.dict{'basis_1': 0, 'basis_2': 'sin omega*x', 'basis_3': 'cos omega*x'}
                2.numbers=3
    :return:    left_matirx:100*numbers
                symbol_matrix:numbers*1
    '''
    x = sp.symbols('x')
    symbol_matrix = pick_basis_function(dict,basis_num=numbers)
    funcs = [sp.lambdify(x, symbol_matrix[i]) for i in range(numbers)]
    left_matirx= torch.zeros(100,numbers,
                                    dtype=torch.float64, device='cuda')
    #generate the 100*numbers
    t = np.linspace(0,2,100)

    for i, func in enumerate(funcs):
        value = func(t)
        value = torch.tensor(value, dtype=torch.float64)
        left_matirx[:, i] = value
    # return the left matrix and
    return left_matirx,symbol_matrix


def convert_data(batch_data:torch.tensor,label:torch.tensor):
    '''
     func:convert the data :1.pick the condition_data
                            2.calcul the gradient
     3                      3.label of txt number
     input: batch_data:[batch,100,9]
            label:[batch]
     :return:    trans_condition[batch,101,2]
                 data_t[batch,100,1]
                 dict_str_solu[batch]
    '''
    device="cuda"
    #label is the csv name
    dict_str_solu=uf.read_real_str(label)
    data_t = batch_data[:, :, 6].detach()
    data_t = data_t.unsqueeze(dim=2)  # shape:[batch,100,1]
    data_t = data_t.to(device).requires_grad_(True)

    # condition  is the first 6 dimen but we need to tranform
    # initial condition[batch,1,2]
    # gradint condition[batch,100,2]
    # condition=cat initial and gradient=[batch,101,2]
    condition = batch_data[:, :, 0:6].to(device)
    ini_condi = condition[:, 0:1, 0:2]
    # critic for real data : z1_t z2_t
    real = batch_data[:, :, 7:9].to(device).requires_grad_(True)  # [batch,100,2]
    # only for diffentiable
    real_data4grad = real.detach()
    real_grads = uf.calculate_diff_grads(real_data4grad, data_t, type="center_diff")
    real_grads = real_grads.to(device)
    real_grads.requires_grad_ = True
    #condition[batch,101,2]
    trans_condition = torch.cat((ini_condi, real_grads), dim=1)
    trans_condition.requires_grad_=True

    return trans_condition,data_t,dict_str_solu

def multiply_matrix(coeff_tensor,omega_tensor,left_matrix,symbol_matrix):
    '''
    input:  coeff_tensor:[batch,basis_num*2],
            omega_tensor:[batch,2]
            left_matrix:[100,3]
            symbol_matrix:[[0][sin_(omegax)][cos_(omegax)]]
    :return: updated_symbol_list[batch*(2)],
             fake_data[batch,100,2],
             fake_condition[batch,101,2],
             basis_matrix[batch,100,basis_num,2]
    '''
    batch_num,basis_nums=coeff_tensor.shape
    single_num=basis_nums//2
    # according to the to update symbol_matrix
    updated_symbol_list = []
    new_functions = []
    data_t= np.linspace(0,2,100)
    z1_left_matirx=torch.zeros((batch_num,100,single_num),requires_grad=True).to("cuda")
    z2_left_matirx = torch.zeros((batch_num, 100, single_num), requires_grad=True).to("cuda")
    z1_gradient=torch.zeros((batch_num,100,1),requires_grad=True).to("cuda")
    z2_gradient = torch.zeros((batch_num, 100,1), requires_grad=True).to("cuda")

    x = sp.Symbol('x')
    #operation for each batch we could output a symbol expression
    for i in range(batch_num):
        updated_symbol_list.append([])
        for symbol in symbol_matrix[:,0]:
            for parameter in omega_tensor[i,:]:
                new_functions.append(symbol.subs(x,parameter*x))

        #get the interval
        z1_new_function=new_functions[::2]  #like [1, -sin(8.5*x), cos(8.5*x)]
        z2_new_function=new_functions[1::2]

        #create the funcs
        z1_funcs = [sp.lambdify(x, z1_new_function[ i]) for i in range(single_num)]
        z2_funcs = [sp.lambdify(x, z2_new_function[i]) for i in range(single_num)]

        #get the expression
        z1_expression = sum([coeff * function for coeff, function in zip(coeff_tensor[i,0:single_num], z1_new_function)])
        z2_expression = sum([coeff * function for coeff, function in zip(coeff_tensor[i,single_num:], z2_new_function)])

        #get the derivation of expression
        grad_z1=sp.diff(z1_expression)
        grad_z2=sp.diff(z2_expression)
        func_grad_z1= sp.lambdify((x), grad_z1, "numpy")
        func_grad_z2 = sp.lambdify((x), grad_z2, "numpy")
        #get gradinet value
        grad_z_value=func_grad_z1(data_t)
        grad_z_value = torch.tensor(grad_z_value, dtype=torch.float64, requires_grad=True).to("cuda")
        z1_gradient[i,:,0] = grad_z_value
        #for z2
        grad_z_value=func_grad_z2(data_t)
        grad_z_value = torch.tensor(grad_z_value, dtype=torch.float64, requires_grad=True).to("cuda")
        z2_gradient[i,:,0] = grad_z_value
        #use the left matirx rather than the analyical expression
        # because the torch cannot support sympy
        for j, func in enumerate(z1_funcs):
            value = func(data_t)
            value = torch.tensor(value, dtype=torch.float64,requires_grad=False).to("cuda")
            z1_left_matirx[i,:, j] = value

        for j, func in enumerate(z2_funcs):
            value = func(data_t)
            value = torch.tensor(value, dtype=torch.float64,requires_grad=False).to("cuda")
            z2_left_matirx[i,:, j] = value

        #update the list
        updated_symbol_list[i].append(z1_expression)
        updated_symbol_list[i].append(z2_expression)
        new_functions=[]

    #calulate the fake data by using batch matrix
    #input1=batch*n*m(batch*100*3). input2=batch*m*p(batch*3*1)
    #output=batch*n*p(bath*100*1)
    coeff_tensor=coeff_tensor.reshape(-1,basis_nums,1)
    #dimension: [100*1]=[100*3]*[3,1]
    fake_z1 = torch.bmm(z1_left_matirx, coeff_tensor[:,0:single_num,0:1])#[batch,100,1]
    fake_z2 = torch.bmm(z2_left_matirx, coeff_tensor[:,single_num:,0:1])#[batch,100,1]

    #fake_data
    fake_data = torch.cat((fake_z1,fake_z2),dim=2)#[batch,100,2]

    fake_ini = fake_data[:,0:1,0:2]
    fake_grad=torch.cat((z1_gradient,z2_gradient),dim=2)#[batch,100,2]

    #fake_condition =cat
    fake_condition=torch.cat((fake_ini,fake_grad),dim=1)

    #basis_matrix =stack
    left_matrix=torch.stack((z1_left_matirx,z2_left_matirx),dim=3)

    return updated_symbol_list, fake_data, fake_condition,left_matrix

import torch.nn.functional as F
# 定义 soft_argmax 和 compute_spectrum 如前所述
def soft_argmax(x, beta=1.0):
    """
        param:  1.x is freq_index [batch,51,1]
                2.beta is a temperature BETA IS LARGER ,THE MORE APPROCH
    """
    x = x * beta
    softmax = F.softmax(x, dim=-1)
    indices = torch.arange(start=0, end=x.size()[-1], dtype=torch.float32).to(x.device)
    return torch.sum(softmax * indices, dim=-1)
import matplotlib.pyplot as plt
import numpy as np
def compute_spectrum(tensor, sampling_rate=50, num_samples=100, device="cuda",argmax=1):

    '''
    Input:
            :param tensor: [batch,100,2]
            :param sampling_rate: 50 hz
            :note =omega/2*pi
    return: omega= resolution * freq_index*2pi:soft_argmax [batch,2]
    '''

    batch,_,vari_dimension = tensor.size()
    soft_freq_index = torch.zeros((batch,vari_dimension), requires_grad=True).to(device)

    for i in range(vari_dimension):

        #fft & P_freqs(positive freqs)
        spectrum = torch.fft.rfft(tensor[:,:,i]) #return [batch，51，2]

        #note： the P_freqs has 0 hz，and resolution=50/100=0.5hz
        P_freqs = torch.fft.rfftfreq(num_samples, d=1 / sampling_rate)[:sampling_rate+1].to(device)#[batch，51，2]

        #magnitude abs
        magnitude = torch.abs(spectrum)  #[batch,51]

        #soft_argmax
        soft_argmax_freq_index = soft_argmax(magnitude)#[batch,1]
        soft_freq_index[:,i] = soft_argmax_freq_index

    soft_omega=soft_freq_index*0.5*2*torch.pi

    return soft_omega


if __name__=="__main__":
    a={'basis_1': 0, 'basis_2': 'sin', 'basis_3': 'cos'}
    left_matirx,symbol_matrix=Get_basis_function_info(a,numbers=3)

