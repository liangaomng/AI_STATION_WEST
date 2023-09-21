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


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output


class unsupervised_omega_generator(nn.Module):
    def   __init__(self,input_dim=202,output_dim=2):
        super(unsupervised_omega_generator, self).__init__()

        self.siren1=Siren(in_features=202, hidden_features=512, hidden_layers=3, out_features=2,
                          outermost_linear=True,
                         first_omega_0=30, hidden_omega_0=30)

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
        self.omega=self.siren1(x)
        #discrete
        #self.omega=RoundWithPrecisionSTE.apply(self.omega,1)

        return self.omega


class omega_generator(nn.Module):
    def __init__(self,input_dim=400,output_dim=2):
        super(omega_generator, self).__init__()
        self.rational_function_1=Rational()
        self.rational_function_2=Rational()


        self.omega_fc1=nn.Sequential(
            nn.Linear(input_dim,512),
            nn.BatchNorm1d(512),
            self.rational_function_1,
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            self.rational_function_2,
            nn.Linear(256,output_dim),
        )
        # kaiming_init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')

    def forward(self,x):
        '''
        :param x: [batch,200,2] ini &gradinets 100*2
        :return: [batch,1]
        '''

        x=x.view(-1,400)

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
    def __init__(self,input_dim=400,output_dim=2):

        super(Generator, self).__init__()
        hidden_width=512
        self.rational_function_1=Rational()
        self.rational_function_2=Rational()
        #structure of model_coeff
        self.model_coeff = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.BatchNorm1d(hidden_width),
            self.rational_function_1,
            nn.Linear(hidden_width, 256),
            nn.BatchNorm1d(256),
            self.rational_function_2,
            nn.Linear(256, output_dim), #2 is the z1 and z2 in 9.12 we add phase1
            #z1=c0+c1*sin（C2t+C3），z2=c4+c5*sin（C6t+C7）

        )
        self.siren2 = Siren(in_features=input_dim,
                            hidden_features=512, hidden_layers=3, out_features=output_dim,
                            outermost_linear=True,
                            first_omega_0=30, hidden_omega_0=30)

        # kaiming_init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        #self.fake_data=torch.zeros(config["batch_size"],100,2,device="cuda").requires_grad_(True)

    def forward(self,x):
        '''
        input:      x is noise [batch,noise_dimension],
                    condition is[batch,101,2]
                    omega_value is [batch,2]
        output:     fake_coeffs:[batch,basis_num*2]
        '''
        #condition is [batch,200,2]
        x=x.view(-1,400)
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

        self.siren2=Siren(in_features=input_dim, hidden_features=512, hidden_layers=3,
                          out_features=1,
                          outermost_linear=True,
                         first_omega_0=30, hidden_omega_0=30)

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

    # initial condition[batch,1,2]
    # gradint condition[batch,100,2]
    # condition=cat initial and gradient=[batch,101,2]
    ini_condi = real_data[:, 0:1, 0:2]

    # critic for real data : z1_t z2_t
    # only for diffentiable
    real_data4grad = real_data.detach()
    real_grads = uf.calculate_diff_grads(real_data4grad, data_t, type="center_diff",plt_show=False)
    real_grads = real_grads.to(device)
    #condition[batch,200,2]
    trans_condition = torch.cat((real_data, real_grads), dim=1)

    return trans_condition,dict_str_solu

from concurrent.futures import ProcessPoolExecutor


def compute_expression_for_batch(i,omega_batch, coeff_batch, data_t, symbol_matrix):
    new_functions=[]
    x = sp.symbols('x')
    psi = sp.symbols('psi')

    column,vari_=7,2
    rows=100
    z1_left_matirx=np.zeros((rows,column))
    z2_left_matirx = np.zeros((rows,column))
    z1_gradient=np.zeros((rows,1))
    z2_gradient = np.zeros((rows, 1))
    grad_batch=0
    left_matrix_batch=0
    for symbol in symbol_matrix[:, 0]:
        if symbol ==1:
            new_functions.append(1)
        else:
            for parameter in omega_batch[:]:
                new_functions.append(symbol.subs(x, parameter*x))


    # get the interval
    z1_new_function = new_functions[0::2]  # like [1, -sin(8.5*x+psi), cos(8.5*x+psi)]
    z2_new_function = new_functions[1::2]

    z2_new_function.insert(0,new_functions[0])# like [1, -sin(8.5*x+psi), cos(8.5*x+psi)]


    # substitute the psi
    z1_new_funcs = [func.subs(psi,0) if hasattr(func, 'subs') else func for func in z1_new_function]
    z2_new_funcs = [func.subs(psi,0) if hasattr(func, 'subs') else func for func in z2_new_function]# no constant

    # get the expression
    z1_expression = sum([coeff * function for coeff, function in zip(coeff_batch[ :,0],
                                                                     z1_new_funcs)])
    z2_expression = sum(
        [coeff * function for coeff, function in zip(coeff_batch[ :,1],
                                                     z2_new_funcs)])
    # update the list
    # updated_symbol_list.append(z1_expression)
    # updated_symbol_list.append(z2_expression)

    # get the derivation of expression
    grad_z1 = sp.diff(z1_expression, x)
    grad_z2 = sp.diff(z2_expression, x)

    # get the function of grad_z1 and grad_z2
    func_grad_z1 = sp.lambdify((x), grad_z1, "numpy")
    func_grad_z2 = sp.lambdify((x), grad_z2, "numpy")

    # get gradinet value
    grad_z_value = func_grad_z1(data_t)

    z1_gradient[:, 0] = grad_z_value

    # for z2
    grad_z_value = func_grad_z2(data_t)

    z2_gradient[ :, 0] = grad_z_value

    # use the left matirx rather than the analyical expression
    # because the torch cannot support sympy
    for j, expr in enumerate(z1_new_funcs):

        if expr == 1:
            value = coeff_batch[0,0]*np.ones(100, dtype=np.float64)
        else:
            func = sp.lambdify((x), expr, "numpy")
            value = func(data_t)

        z1_left_matirx[ :, j] = value

    for j, expr in enumerate(z2_new_funcs):

        if expr == 1:
            value = coeff_batch[0,1]*np.ones(100, dtype=np.float64)
        else:
            func = sp.lambdify((x), expr, "numpy")
            value = func(data_t)


        z2_left_matirx[:, j] = value





    return z1_left_matirx,z2_left_matirx,z1_gradient,z2_gradient


import time


def generate_new_functions(symbol_matrix, omega_tensor_batch):
    new_functions = []
    x = sp.Symbol('x')
    for symbol in symbol_matrix[:, 0]:
        if symbol == 1:
            new_functions.append(1)
        else:
            for parameter in omega_tensor_batch:
                new_functions.append(symbol.subs(x, parameter * x))
    return new_functions


def get_function_expressions(coeff_tensor_batch, new_functions):
    psi = sp.Symbol('psi')
    z1_new_functions = [func.subs(psi, 0) if hasattr(func, 'subs') else func for func in new_functions[0::2]]
    z2_new_functions = [func.subs(psi, 0) if hasattr(func, 'subs') else func for func in new_functions[1::2]]
    z2_new_functions.insert(0, new_functions[0])
    z1_expression = sum([coeff * func for coeff, func in zip(coeff_tensor_batch[:, 0], z1_new_functions)])
    z2_expression = sum([coeff * func for coeff, func in zip(coeff_tensor_batch[:, 1], z2_new_functions)])
    return z2_new_functions,z2_new_functions,z1_expression, z2_expression


def get_gradient(expression):
    x = sp.Symbol('x')
    gradient = sp.diff(expression, x)
    return sp.lambdify(x, gradient, "numpy")


def get_left_matrix_values(expression, coeff_tensor_batch, data_t):
    if expression == 1:
        value = coeff_tensor_batch[0] * torch.ones(100, dtype=torch.float64).to("cuda")
    else:
        func = sp.lambdify(sp.Symbol('x'), expression, "numpy")
        value = torch.tensor(func(data_t), dtype=torch.float64).to("cuda")
    return value


def return_torch_version_matrix(coeff_tensor, omega_tensor, symbol_matrix):
    '''

    :param coeff_tensor: [batch,coeff_number,vari]
    :param omega_tensor:
    :param symbol_matrix: [[1],[sin]...]
    :return:
    '''
    batch_num, coeff_number, vari = coeff_tensor.shape
    _, freq_num, _ = omega_tensor.shape
    basis_nums = coeff_number  #
    # according to the to update symbol_matrix
    updated_symbol_list_z1 = []
    updated_symbol_list_z2 = []
    new_functions = []
    data_t = torch.linspace(0, 2, 100).unsqueeze(0).unsqueeze(2).to("cuda")  # shape (1, 100, 1). to("cuda")
    # get the left_matirx
    #[32,100,7]
    z1_left_matirx = torch.zeros((batch_num, 100, basis_nums), requires_grad=True).to("cuda")
    z2_left_matirx = torch.zeros((batch_num, 100, basis_nums), requires_grad=True).to("cuda")
    omega_z1= omega_tensor[:,:,0]  # Shape is (32, 3)
    omega_z2 = omega_tensor[:, :, 1]  # Shape is (32, 3)

    for i,exprs in enumerate(symbol_matrix[:, 0]):


        if exprs==1:
            #batch multiply
            a = torch.ones((batch_num,100),dtype=torch.float64).to("cuda") * coeff_tensor[:, 0, 0].unsqueeze(1)
            b = torch.ones((batch_num,100),dtype=torch.float64).to("cuda") * coeff_tensor[:, 0, 1].unsqueeze(1)
            z1_left_matirx[:,:,0] = a
            z2_left_matirx[:,:,0] = b

        elif isinstance(exprs, sp.sin):

            #[batch,100,freq_numbers]
            z1 = torch.sin(omega_z1.unsqueeze(1) * data_t)  # Broadcasting is done here
            z1_left_matirx[:,:,1:freq_num+1] = z1
            z2 = torch.sin(omega_z2.unsqueeze(1) * data_t)  # Broadcasting is done here
            z2_left_matirx[:,:,1:freq_num+1] = z2

        elif isinstance(exprs, sp.cos):
            #[batch,100,freq_numbers]
            z1 = torch.cos(omega_z1.unsqueeze(1)  * data_t)  # Broadcasting is done here
            z1_left_matirx[:,:,freq_num+1:] = z1
            z2 = torch.sin(omega_z2.unsqueeze(1) * data_t)  # Broadcasting is done here
            z2_left_matirx[:,:,freq_num+1:] = z2

    left_matrix=torch.stack((z1_left_matirx,z2_left_matirx),dim=3)#[32, 100, 7, 2]

    fake_z1 = torch.bmm(left_matrix[:,:,:,0], coeff_tensor[:,:,0:1])  # return [batch,100,1]
    fake_z2 = torch.bmm(left_matrix[:,:,:,1], coeff_tensor[:,:,1:2])  # return [batch,100,1]
    fake_data=torch.cat((fake_z1,fake_z2),dim=2)
    fake_gradient=uf.calculate_diff_grads(fake_data,data_t,plt_show=False)
    fake_condition = torch.cat((fake_data, fake_gradient), dim=1)  # [batch,200,2]

    return dict,left_matrix,fake_data,fake_condition


import torch.nn.functional as F
#  soft_argmax & compute_spectrum
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
def compute_spectrum(tensor, sampling_rate=49.5, num_samples=100, device="cuda",
                     beta=1,
                     freq_number=1,
                     train_step=0,path=0):

    '''
    Input:
            :param tensor: [batch,100,2]
            :param sampling_rate: 49.5 hz
            :param freq_number: 1
            :note =omega/2*pi
            :train_step
            :every epoch save the spectrum

    return: omega= resolution * freq_index*2pi:soft_argmax [batch,2]
    '''

    batch,_,vari_dimension = tensor.size()
    resolution=sampling_rate/num_samples
    soft_freq_index = torch.zeros((batch,3,vari_dimension), requires_grad=True).to(device)
    training_omage = {}
    # note： the P_freqs has 0 hz，and resolution=2/99=49.5hz
    P_freqs = torch.fft.rfftfreq(num_samples, d=1 / sampling_rate)[:].to(device)  # [batch，50，2]
    for i in range(vari_dimension):

        #fft & P_freqs(positive freqs)
        spectrum = torch.fft.rfft(tensor[:,:,i]) #return [batch，51，2]

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
        #save for plot
        if train_step % 256 ==0:
            epoch_omega=train_step/256
            training_omage={"raw_data":tensor,
                       "P_freqs":P_freqs,
                       "soft_freq_index":soft_freq_index,
                       "epoch":epoch_omega,
                     }
            torch.save(training_omage, path+f"{int(epoch_omega)}.pth")


    soft_omega=soft_freq_index*resolution*2*torch.pi

    return soft_omega
def compute_sequence(fake_coeffs,generator_freq,dict):
    '''
    :param fake_coeffs: [batch,7,2]
    :param generator_freq: [batch,3,2]
    :param str: dict{"basis_1": "x**0", "basis_2": "sin", "basis_3": "cos"}
    :return: [batch,2]
    '''
    batch,_,vari_dimension = fake_coeffs.size()
    sequence = torch.zeros((batch,vari_dimension), requires_grad=True).to("cuda")




    return sequence





if __name__=="__main__":
    a={'basis_1': 0, 'basis_2': 'sin', 'basis_3': 'cos'}
    left_matirx,symbol_matrix=Get_basis_function_info(a,numbers=3)

