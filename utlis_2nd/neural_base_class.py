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

class TemperatureSoftmax(nn.Module):
    def __init__(self, temperature=1.0,soft_dim=1):
        super(TemperatureSoftmax, self).__init__()
        self.temperature = temperature
        self.dim = soft_dim

    def forward(self, logits):

        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=self.dim)
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
    def forward(ctx, input, decimal_places=1):
        multiplier = 10 ** decimal_places
        return torch.round(input * multiplier) / multiplier

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # identity gradient

from torch.autograd import Variable
class gumble_softmax(nn.Module):
    def __init__(self,device='cpu'):
        super(gumble_softmax, self).__init__()
        self.laten_dim=2
        self.categorical_dim=51
        self.device=device
    def sample_gumbel(self,shape, eps=1e-20):
        U = torch.rand(shape).cpu()
        return Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self,logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self,logits, temperature):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(-1, self.latent_dim * self.categorical_dim)

temp_softmax=TemperatureSoftmax()

from collections import namedtuple

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
# 创建一个 Transition 对象
trans = Transition(state=1, action=2, reward=3, next_state=4, done=False)
class Omgea_MLPwith_residual_dict(nn.Module):

    def __init__(self,
                 input_sample_lenth,
                 hidden_dims,
                 output_coeff=False,
                 hidden_act='rational',
                 output_act='softmax',
                 sample_vesting=2,#unit （s）
                 grad_order=0,#1 order default
                 vari_number=2,
                 Combination_basis=None,
                 device_type="cuda",
                 sample_model='topk',
                 pre_process="None",

                 soft_temp=["False",1.0],
                 distll_temp=1.0,
                 learn_distill_temp=False,
                 prob_sample_numb=1,
                 gumble_temp=1.0,
                 Sample_choice=False,
                 dropout=0.1,
                 ):

        super(Omgea_MLPwith_residual_dict, self).__init__()
        if Combination_basis is None:
            self.basis_function = ["1","sin", "cos"]
        self.device_type=device_type

        self.sample_model=sample_model
        self.pre_process=pre_process
        self.prob_sample_numb=prob_sample_numb
        self.Sample_choice=Sample_choice


        if(soft_temp[0]==False):
            #can upgrade
            self.register_parameter("softarg_temp",torch.nn.Parameter(torch.tensor(soft_temp)))
        else:
            #cannot upgrade
            self.register_buffer("softarg_temp",torch.nn.Parameter(torch.tensor(soft_temp)))

        if(learn_distill_temp==False):
            #can upgrade
            self.register_parameter("distill_temp",torch.nn.Parameter(torch.tensor(distll_temp)))
        else:
            #cannot upgrade
            self.register_buffer("distill_temp",torch.nn.Parameter(torch.tensor(distll_temp)))


        self.register_buffer('buffer_sample_time', torch.tensor(sample_vesting))
        self.register_buffer('buffer_sample_length', torch.tensor(0)) #100
        self.register_buffer('buffer_sample_rate', torch.tensor(0))#2s
        self.register_buffer('buffer_delta_t', torch.tensor(grad_order))
        self.register_buffer('buffer_freq_index', torch.tensor(0))#[0,...,50]
        self.register_buffer('buffer_freq_numbers', torch.tensor(0))#0.02
        self.register_buffer('buffer_vari_number', torch.tensor(vari_number))


        # index of freq and numbers
        self.buffer_sample_length = torch.tensor(input_sample_lenth)
        self.buffer_delta_t = self.buffer_sample_time/(self.buffer_sample_length-1)
        self.buffer_sample_rate = torch.reciprocal(self.buffer_delta_t)  #49.5h
        self.buffer_freq_index = torch.fft.rfftfreq(self.buffer_sample_length, d=self.buffer_delta_t)
        self.buffer_freq_numbers = torch.tensor(self.buffer_freq_index.shape[0])

        if grad_order==1:
            input_dim = self.buffer_sample_length*2*grad_order*vari_number
        elif grad_order==0:
            input_dim = self.buffer_sample_length*vari_number

        output_dim = self.buffer_freq_numbers*vari_number

        if output_coeff== True: #4inference net

            self.non_zero_freq_num=self.buffer_freq_index[1:].numel()

            output_dim = vari_number* (2*self.non_zero_freq_num+1) #2 means that sin and cos family
            print("output_dim",output_dim)

        self.layers = nn.ModuleDict({
            "layer_norm": nn.LayerNorm([self.buffer_sample_length*vari_number]),# feature=[100*3]
            'input': nn.Linear(input_dim, hidden_dims[0]),
            'hidden': nn.ModuleList(),
            'output': nn.Linear(hidden_dims[-1],output_dim),
            'hidden_act': nn.ModuleList(),

        })

        self.dropout = nn.Dropout(dropout)


        if output_act == 'softmax':
            self.output_act = nn.Softmax(dim=1)
        elif output_act == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_act == 'Identity':
            print("output_act is Identity")
            self.output_act = nn.Identity()
        elif output_act == 'round':
            self.output_act = RoundWithPrecisionSTE.apply()
        elif output_act == "distill_temp":
            self.output_act = TemperatureSoftmax(self.softarg_temp)


        prev_dim=hidden_dims[0]

        for i, dim in enumerate(hidden_dims[0:]):

            self.layers['hidden'].append(nn.Linear(prev_dim, dim))
            if hidden_act == 'rational':

                self.layers['hidden_act'].append(Rational())


            elif hidden_act == 'learnable_phi':

                self.layers['hidden_act'].append(phi_learnable_act())

            elif hidden_act=='phi_b_line':

                self.layers['hidden_act'].append(phi_b_line())

            prev_dim = dim

    def convert_data_2_cat_grad(self,tensor):
        '''
        :param tensor: [batch,sample_length,vari_number]
        :return: [batch,order,sample_length,vari_number]
        '''
        batch,length,vari_number=tensor.shape

        #calculate gradient
        grad_tensor=uf.five_point_stencil(data=tensor,dt= self.buffer_delta_t)
        grad_tensor=grad_tensor.unsqueeze(dim=1) # [batch,1,sample_length,vari_number]
        tensor=tensor.unsqueeze(dim=1) # [batch,1,sample_length,vari_number]
        cat_tensor=torch.cat([tensor,grad_tensor],dim=1)   # '''先不考虑噪声'''
        return cat_tensor

    def return_fft_spectrum(self,tensor,
                            need_norm=True,vari_numbers=2):
        '''

        :param tensor: [batch,t_setp,vari_number],like [256,100,3]

        :return: [batch,magn,vari_number],like [256,51,3]
        '''
        batch,t_set,vari_number=tensor.shape
        #pick one vari
        pick=tensor[:, :, 0:vari_numbers]
        #fft
        fft_tensor=torch.fft.rfft(pick,n=self.buffer_sample_length.item(),
                                  dim=1)
        #abs
        magn_tensor=torch.abs(fft_tensor) #[batch,51,3]

        if need_norm:
            norm_magn_tensor=F.normalize(magn_tensor, p=1.0, dim=1)

            return norm_magn_tensor
        else:

            return magn_tensor
    def Return_sample_freq_index(self,
                                 freq_distrubtion_tensor,
                                 model="soft_argmax",
                                 pre_process="None",
                                 temp=1.0) -> torch.Tensor:
        '''
        :param
             1.model = "topk" top from
             2.model  ="adaptive topk" top from
             3.model ="multi-nomial"

        :return:
        '''
        # sample function from fouier space
        batch_num,freq,vari=freq_distrubtion_tensor.shape

        sample_index_vari = torch.zeros(batch_num, self.prob_sample_numb, vari).to(self.device_type)


        if pre_process == "gumble":

            gumble_tensor = torch.zeros(batch_num, freq, vari).to(self.device_type)

            for i in range(vari):
                gumble_tensor[:, :, i] = F.gumbel_softmax(freq_distrubtion_tensor[:, :, i],
                                                          tau=self.gumble_temp,
                                                          hard=False,
                                                          eps=1e-15)  # [256,51,2]

            process_tensor=gumble_tensor

        elif pre_process == "None":

            process_tensor=freq_distrubtion_tensor

        if model == "topk":

            value,sample_index_vari = torch.topk(process_tensor[:,1:,:],dim=1,k=self.prob_sample_numb)

        elif model == "multi-nominal":


            for i in range(vari):
                vari_sample_index= torch.multinomial(process_tensor[:,:,i],
                                                     self.prob_sample_numb,
                                                     replacement=False)
                sample_index_vari[:,:,i]=vari_sample_index

        elif model == "soft_argmax":

            tau=self.temp

            soft_prob = F.softmax(process_tensor / tau, dim=1).requires_grad_(True)

            # soft argmax
            indices = self.buffer_freq_index.view(1, freq, 1).expand(batch_num, freq, 2)

            soft_index = torch.sum(soft_prob * indices ,dim=1)
            sample_index_vari=soft_index.unsqueeze(1)


        sample_index_vari=sample_index_vari.long()

        # return to tensor
        return sample_index_vari

    def Return_omega_vari_tensor(self,
                                 batch_num,
                                 Sample,
                                 Sample_index=None):

        if Sample==False:
            # 找到非零索引
            nonzero_indices = torch.nonzero(self.buffer_freq_index)

            # 使用非零索引获得非零张量
            nonzero_tensor = self.buffer_freq_index[nonzero_indices]

            # 转换到适当的设备
            nonzero_tensor = nonzero_tensor.to(self.device_type)

            # 扩展并重塑张量
            omega_value_var = nonzero_tensor.unsqueeze(0).repeat(batch_num, 1, 1)
            omega_value_var = omega_value_var.reshape(batch_num, 1, self.non_zero_freq_num)

            # 由于 omega_value_var1 和 omega_value_var2 是相同的，只需创建一个
            omega_value_var1 = omega_value_var
            omega_value_var2 = omega_value_var

            return omega_value_var1,omega_value_var2

        else:

             #Sample_index should be [256,12,2]

             # 扩展和重塑 buffer_freq_index
             omega_value_var = self.buffer_freq_index.unsqueeze(0).to(self.device_type)
             omega_value_var = omega_value_var.repeat(batch_num, 1)
             # 注意：omega_value_var 的形状已经是 [batch_num, 51]，因此不需要重塑

             # 根据 Sample_index 提取样本
             omega_value_var1_sample = torch.gather(omega_value_var, 1, index=Sample_index[:, :, 0])
             omega_value_var1_sample = omega_value_var1_sample.unsqueeze(dim=1)

             omega_value_var2_sample = torch.gather(omega_value_var, 1, index=Sample_index[:, :, 1])
             omega_value_var2_sample = omega_value_var2_sample.unsqueeze(dim=1)

             return omega_value_var1_sample, omega_value_var2_sample

    def return_pred_data(self,coeff_tensor,
                         freq_distrubtion_tensor):
        '''
        :param coeff_tensor: [batch,51,vari_number]--trainable
         hard_mean :the max numbers is very big, but we could regulization it
        :return: [batch,t_step,vari_number]
        '''
        # get the sample

        prior_knowledge_matrix=["1","sin","cos"]
        batch_num, coeff_number, vari = coeff_tensor.shape

        _, freq_num, _ = freq_distrubtion_tensor.shape

        prior_omega_knowledge_number=len(prior_knowledge_matrix)-1


        # according to the to update symbol_matrix

        data_t = torch.linspace(0, self.buffer_sample_time, self.buffer_sample_length).\
            unsqueeze(0).unsqueeze(2).to(self.device_type)  # shape (1, 100, 1). to("cuda")

        if self.Sample_choice==True:

            basis_nums = prior_omega_knowledge_number * (self.prob_sample_numb) + 1  # 2*50+dc=101# like [32,100,12] #+1 is dc

            z1_left_matirx = torch.zeros((batch_num,
                                          self.buffer_sample_length, basis_nums), requires_grad=False).to( self.device_type)
            z2_left_matirx = torch.zeros((batch_num,
                                          self.buffer_sample_length, basis_nums), requires_grad=False).to( self.device_type) #like [32,100,101]

            Sample_omega_index = self.Return_sample_freq_index(freq_distrubtion_tensor,
                                                               model=self.sample_model,
                                                               pre_process=self.pre_process,
                                                               temp=self.soft_temp,
                                                               ) # like [256,12,2] 2 is vari



            omega_value_var1, omega_value_var2 = self.Return_omega_vari_tensor(batch_num=batch_num,
                                                                               Sample=self.Sample_choice,
                                                                               Sample_index=Sample_omega_index) # like [256,100,12]



        elif self.Sample_choice==False:

            basis_nums = prior_omega_knowledge_number * (self.non_zero_freq_num) + 1  # 2*50+dc=101

            z1_left_matirx = torch.zeros((batch_num,
                                          self.buffer_sample_length, basis_nums), requires_grad=False).to(self.device_type)
            z2_left_matirx = torch.zeros((batch_num,
                                          self.buffer_sample_length, basis_nums), requires_grad=False).to(self.device_type)  # like [32,100,101]

            omega_value_var1, omega_value_var2 = self.Return_omega_vari_tensor(batch_num=batch_num,
                                                                               Sample=self.Sample_choice) # like [256,12]

            Sample_omega_index= torch.arange(start=0, end=50, step=1)

            Sample_omega_index =  Sample_omega_index.reshape(1,50,1)


            Sample_omega_index=Sample_omega_index.repeat(batch_num,1,2)# like[256, 50, 2]
            Sample_omega_index = Sample_omega_index.long()


            self.prob_sample_numb = (basis_nums - 1) // prior_omega_knowledge_number



        for i, exprs in enumerate(prior_knowledge_matrix):

            if exprs == "1":#have set ones
                z1_left_matirx[:, :, 0] = 1
                z2_left_matirx[:, :, 0] = 1

            if exprs == "sin":
                # [batch,100,freq_numbers]  omega_z1.unsqueeze(1) is [batch,1,freq_numbers]
                z1 = torch.sin(omega_value_var1 * data_t)  # Broadcasting is done here

                z1_left_matirx[:, :, 1:self.prob_sample_numb+1] = z1

                z2 = torch.sin(omega_value_var2 *data_t)  # Broadcasting is done here

                z2_left_matirx[:, :, 1:self.prob_sample_numb+1] = z2

            elif  exprs == "cos":
                # [batch,100,freq_numbers]
                z1 = torch.cos(omega_value_var1* data_t)  # Broadcasting is done here
                z1_left_matirx[:, :, self.prob_sample_numb+1:] = z1
                z2 = torch.cos(omega_value_var2 * data_t)  # Broadcasting is done here
                z2_left_matirx[:, :, self.prob_sample_numb+1:] = z2

        left_matrix = torch.stack((z1_left_matirx, z2_left_matirx), dim=3)  # [batch, t_steps, freq_num, vari]

        #coeff_tensor[batch,101,2]
        Coeff_sample_tensor=coeff_tensor[:,1:,:].reshape(-1,self.non_zero_freq_num,prior_omega_knowledge_number,vari)

        Sample_omega_index_pick=Sample_omega_index.unsqueeze(2) # mark


        Sample_omega_index_pick=Sample_omega_index_pick.repeat(1,1,prior_omega_knowledge_number,1)

        Coeff_sample_tensor= torch.gather(Coeff_sample_tensor,dim=1,index=Sample_omega_index_pick) # [batch,50,2,2]

        Coeff_sample_tensor=Coeff_sample_tensor.reshape(-1,self.prob_sample_numb*prior_omega_knowledge_number,vari)

        Cat_dc_sample= torch.cat((coeff_tensor[:,0:1,:],Coeff_sample_tensor),dim=1) # [batch,101,2]


        # [batch, t_steps, freq_num, vari]*[batch, freq_num, vari,1]=[batch,t_steps,vari,1]
        pred_1 = torch.bmm(left_matrix[:, :, :, 0], Cat_dc_sample[:, :, 0:1])  # return [batch,t_steps,1]
        pred_2 = torch.bmm(left_matrix[:, :, :, 1], Cat_dc_sample[:, :, 1:2])  # return [batch,t_steps,1]

        pred_tensor = torch.cat((pred_1, pred_2), dim=2)  # [batch,100,2]

        self.register_buffer("left_matrix", left_matrix)

        return left_matrix,pred_tensor

    def forward(self, x):

        '''
        suppose we have time series data with 100 time steps in two variables
        after convert data to [batch,t_step,vari_number],channel is adding the one order-derivatiation to dim=1
        :param x: [batch,t_step,vari_number]
        eg: return [batch,freq_num,vari_number]
        '''
        batch,t_step,vari_number=x.shape
        x=x.reshape(batch,-1) #[batch,100*2*3]

        x=self.layers["layer_norm"](x)
        x = self.layers['input'](x)
        x = self.layers['hidden_act'][0](x)
        residual = x

        for i,(layer,act) in enumerate(zip(self.layers['hidden'][0:-1],self.layers['hidden_act'][0:-1])):
            x = layer(x)
            x= self.dropout(x)
            x = act(x)
            x += residual #resiual connection
            residual = x
        x=self.layers['hidden'][-1](x)
        x = self.layers['output'](x)
        #for varis softmax
        x=x.view(batch,-1,vari_number)
        # OMEGA is soft-to prob distribution [batch,51,vari_number]
        # inference is to get the coefficient by the prior knowledge

        x = self.output_act(x)


        return x

    def calculate_entropy(self,prob_distributions,dim_default=1):
        '''
            input:prob_distributions:[batch,51,vari_number]
            return [batch,scalar]
        '''
        print("input",prob_distributions.shape)
        # ensure the sum of prob is 1
        prob_distributions = F.normalize(prob_distributions, p=1.0, dim=1)
        plt.plot(prob_distributions.squeeze(0).detach().numpy())
        plt.show()

        # cal entropy scalar
        # 1e-10 to prevent log(0)
        entropies = -torch.sum(prob_distributions * torch.log2(prob_distributions + 1e-9), dim=dim_default)

        return entropies

    def save_tensor4visual(self,**kwargs):
        '''
        this is a function for saving the tensor for visualization in the path
        kwargs:
                1.dict_tensor
                2.path
                3.epoch_times
        '''
        dict_tensor=kwargs["save_dict"]
        epoch_omega=kwargs["epoch_record"]
        filepath=kwargs["path"]

        torch.save(dict_tensor,filepath+"/"+"tensor_"+f"{int(epoch_omega)}.pth")


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


def convert_data(real_data:torch.tensor,data_t,label:torch.tensor,eval=False):
    '''
     func:convert the data :1.pick the condition_data
                            2.calcul the gradient
                            3.label of txt number
     input: real_data:[batch,100,2]
            label:[batch]
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
def compute_spectrum_normlized(
                     data_tensor,
                     sampling_rate=49.5,
                     domin_number=32,
                     train_step=0,filepath=0,name="",
                     label_save=0):


    """
    Perform FFT and return normalized frequency and magnitude.

    Parameters:
    - input_tensor: PyTorch tensor of shape [batch_size, time_steps, num_variables]

    Returns:
    - normalized_magnitude: Normalized magnitude for each frequency, shape [batch_size, num_frequencies, num_variables]
    - freqs: Frequencies corresponding to each FFT output, shape [num_frequencies]
    """
    batch_size, time_steps, num_variables = data_tensor.shape

    # Perform FFT
    spectrum = torch.fft.rfft(data_tensor, dim=1)  # Shape: [batch_size, num_frequencies, num_variables, 2]

    # Compute magnitude
    magnitude = torch.abs(spectrum)  # Shape: [batch_size, num_frequencies, num_variables]

    # Normalize
    max_magnitude = torch.max(magnitude, dim=1, keepdim=True)[0]  # Shape: [batch_size, 1, num_variables]

    normalized_magnitude = magnitude / max_magnitude  # Shape: [batch_size, num_frequencies, num_variables]

    # Compute frequencies
    freqs = torch.fft.rfftfreq(time_steps,d=1/sampling_rate)  # Shape: [num_frequencies]

    #save for plot
    if train_step % domin_number ==0:

        epoch_omega=train_step/domin_number
        info={
                    "raw_data":data_tensor,
                    "P_freqs":freqs,
                    "epoch":epoch_omega,
                    "label_save":label_save
                 }

        torch.save(info, filepath+"/"+name+"_"+f"{int(epoch_omega)}.pth")

    return normalized_magnitude, freqs




