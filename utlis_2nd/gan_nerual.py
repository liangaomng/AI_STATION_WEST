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


class Generator(nn.Module):

    pass




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
    pass




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
    psi=sp.symbols('psi')
    for key, value in dict.items():
        if isinstance(value, str):
            if value == 'sin':
                symbols.append(sp.sin(x+psi))
            elif value == 'cos':
                symbols.append(sp.cos(x+psi))
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
    :return: #Matrix([[0], [sin(x+psi)], [cos(x+psi)]])
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
                #note sympy is according to the rad
                #1rad=180/pi
    '''
    x = sp.symbols('x')
    psi = sp.symbols('psi')
    symbol_matrix = pick_basis_function(dict,basis_num=numbers)

    funcs = [sp.lambdify((x,psi), symbol_matrix[i]) for i in range(numbers)]

    left_matirx= torch.zeros(100,numbers,
                             dtype=torch.float64,
                             device='cuda')
    #generate the 100*numbers
    t = np.linspace(0,2,100)
    psi=0

    for i, func in enumerate(funcs):

        value = func(t,psi)
        value = torch.tensor(value, dtype=torch.float64)
        left_matirx[:, i] = value


    return left_matirx,symbol_matrix


def convert_data(real_data:torch.tensor,data_t,label:torch.tensor,step,eval=False):
    '''
     func:convert the data :1.pick the condition_data
                            2.calcul the gradient
                            3.label of txt number
     input: real_data:[batch,100,2]
            label:[batch]
            step:scalar
            eval:bool,to choose if the function : uf.read_real_str(label)
     :return:    trans_condition[batch,101,2]
                 data_t[batch,100,1]
                 dict_str_solu[batch]
    '''
    device="cuda"
    #label is the csv name ---eval to use!
    if eval:
        dict_str_solu=uf.read_real_str(label)
    else:
        dict_str_solu=0

    # gradint condition[batch,100,2]

    # critic for real data : z1_t z2_t
    # only for diffentiable
    real_data4grad = real_data.detach()
    real_grads = uf.calculate_diff_grads(real_data4grad, data_t, type="center_diff",plt_show=False)
    real_grads = real_grads.to(device)
    #condition[batch,200,2]
    trans_condition = torch.cat((real_data, real_grads), dim=1)

    return trans_condition,dict_str_solu

from concurrent.futures import ProcessPoolExecutor


def get_gradient(expression):
    x = sp.Symbol('x')
    gradient = sp.diff(expression, x)
    return sp.lambdify(x, gradient, "numpy")
import torch.nn.functional as F
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
def compute_spectrum(data_tensor,sampling_rate=49.5,
                     num_samples=100, device="cuda",
                     beta=1,
                     freq_number=1,domin_number=32,
                     train_step=0,filepath=0,name="",label_save=0):

    '''
    Input:
            :param tensor: [batch,100,2]
            :pred_freq_tensor: [batch,freq_number,2]
            :param sampling_rate: 49.5 hz
            :param freq_number: 1
            :note =omega/2*pi

            :train_step
            :every epoch save the spectrum

    return: omega= resolution * freq_index*2pi:soft_argmax [batch,2]
    '''

    batch,_,vari_dimension = data_tensor.size()
    resolution=sampling_rate/num_samples
    soft_freq_index = torch.zeros((batch,freq_number,vari_dimension), requires_grad=True).to(device)

    # note： the P_freqs has 0 hz，and resolution=2/99=49.5hz
    P_freqs = torch.fft.rfftfreq(num_samples, d=1 / sampling_rate)[:].to(device)  # [batch，51]


    for i in range(vari_dimension):

        #fft & P_freqs(positive freqs)
        spectrum = torch.fft.rfft(data_tensor[:,:,i]) #return [batch，51，2]

        #magnitude abs
        magnitude = torch.abs(spectrum)  #[batch,51]

        #soft_argmax
        # soft_argmax_freq_index = soft_argmax(magnitude,beta)#[batch,1]
        # soft_freq_index[:,i] = soft_argmax_freq_index
        for j in range(freq_number):
            soft_idx = soft_argmax(magnitude, beta)  # [batch]
            soft_idx = soft_idx.reshape(batch,1)
            soft_freq_index[:, j:j+1, i] = soft_idx
            # Mask the magnitude by setting the maximum value to a very small value--0
            top_val, top_idx = torch.max(magnitude, dim=1, keepdim=True)
            magnitude.scatter_(1, top_idx,0)

        soft_omega = soft_freq_index * resolution * 2 * torch.pi

    #save for plot
    if train_step % domin_number ==0:

        epoch_omega=train_step/domin_number
        info={
                "raw_data":data_tensor,
                    "P_freqs":P_freqs,
                    "soft_freq_index":soft_freq_index,
                    "soft_omega":soft_omega,
                    "epoch":epoch_omega,
                    "label_save":label_save
                 }

        torch.save(info, filepath+"/"+name+"_"+f"{int(epoch_omega)}.pth")

    soft_omega=soft_freq_index*resolution*2*torch.pi
    return soft_omega




if __name__=="__main__":
    a={'basis_1': 0, 'basis_2': 'sin', 'basis_3': 'cos'}
    left_matirx,symbol_matrix=Get_basis_function_info(a,numbers=3)

