import torch
import torch.nn as nn

import utlis_2nd.utlis_funcs as uf

import sympy as sp
'''
base class for neural_network
'''

class sReLU(torch.nn.Module):
    def __init__(self):
        super(sReLU, self).__init__()

    def forward(self, x):
        return F.relu(-(x - 1)) * F.relu(x)

class phi_b_line(nn.Module):

    def __init__(self):
        super(phi_b_line, self).__init__()

    def forward(self, x):
        return (
                F.relu(x - 0) ** 2
                - 3 * F.relu(x - 1) ** 2
                + 3 * F.relu(x -2) ** 2
                - F.relu(x - 3) ** 2
        )
class phi_learnable_act(nn.Module):
    def __init__(self):
        super(phi_learnable_act, self).__init__()
        # register parameter
        self.alpha = torch.tensor(1.0, requires_grad=True)
        self.register_parameter('alpha', nn.Parameter( self.alpha))

        self.beta = torch.tensor(2.0, requires_grad=True)
        self.register_parameter('beta', nn.Parameter( self.beta))

        self.gamma = torch.tensor(3.0, requires_grad=True)
        self.register_parameter('gamma', nn.Parameter( self.gamma))

        self.coff = torch.tensor([1.0,-3.0,3.0,-1.0], requires_grad=True)
        self.register_parameter('coff', nn.Parameter( self.coff ))


    def forward(self, x):
        return (
                self.coff[0]*F.relu(x - 0) ** 2
                +self.coff[1] * F.relu(x -  self.alpha) ** 2
                + self.coff[2] * F.relu(x - self.beta) ** 2
                +self.coff[3] *F.relu(x -  self.gamma) ** 2
        )
    def show(self):
        #just for visulization
        return self.coff,self.alpha,self.beta,self.gamma


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


class omega_generator(nn.Module):
    def __init__(self,input_dim=400,output_dim=2,act='rational'):
        super(omega_generator, self).__init__()

        if act=='rational':
            self.act1 = Rational()
            self.act2 = Rational()
        elif act=='learnable_phi':
            self.act1 = phi_learnable_act()
            self.act2 = phi_learnable_act()


        self.omega_fc1=nn.Sequential(
            nn.LayerNorm([input_dim]),
            nn.Linear(input_dim,512),
            nn.BatchNorm1d(512),
            self.act1,
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            self.act2,
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

class Coeff_period_solu(nn.Module):
    '''
    Generator has two neural networks
    '''
    def __init__(self,input_dim=400,output_dim=2,act='rational'):

        super(Coeff_period_solu, self).__init__()
        hidden_width=512
        if act=='rational':
            self.act1 = Rational()
            self.act2 = Rational()
        elif act=='learnable_phi':
            self.act1 = phi_learnable_act()
            self.act2 = phi_learnable_act()

        #structure of model_coeff
        self.model_coeff = nn.Sequential(
            nn.LayerNorm([input_dim]),
            nn.Linear(input_dim, hidden_width),
            nn.BatchNorm1d(hidden_width),
            self.act1,
            nn.Linear(hidden_width, 256),
            nn.BatchNorm1d(256),
            self.act2,
            nn.Linear(256, output_dim),
        )


        # kaiming_init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')

    def forward(self,x):
        '''
        input:      x is noise [batch,vari_dim],
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


class Omgea_MLPwith_residual_dict(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim,hidden_act='rational',output_act='softmax'):
        super(Omgea_MLPwith_residual_dict, self).__init__()


        self.layers = nn.ModuleDict({
            'input': nn.Linear(input_dim, hidden_dims[0]),
            'hidden': nn.ModuleList(),
            'output': nn.Linear(hidden_dims[-1], output_dim),

        })

        if output_act == 'softmax':
            self.output_act = nn.Softmax(dim=-1)
        elif output_act == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_act == 'linear':
            self.output_act = nn.Identity()
        elif output_act == 'round':
            self.output_act = RoundWithPrecisionSTE.apply()


        prev_dim=hidden_dims[0]

        for i, dim in enumerate(hidden_dims[1:]):

            self.layers['hidden'].append(nn.Linear(prev_dim, dim))
            if hidden_act == 'rational':
                self.layers['act'].append(Rational())
            elif hidden_act == 'learnable_phi':
                self.layers['act'].append(phi_learnable_act())
            elif hidden_act=='phi_b_line':
                self.layers['act'].append(phi_b_line())
            prev_dim = dim

    def forward(self, x):
        x = self.layers['input'](x)
        x = self.layers['act'][0](x)
        residual = x
        for i,(layer,act) in enumerate(zip(self.layers['hidden'],self.layers['act'][:])):
            x = layer(x)
            x = act(x)
            x += residual #resiual connection
            residual = x

        x = self.layers['output'](x)
        #softmax -to prob
        x = self.output_act(x)
        return x


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


