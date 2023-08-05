'''
hyper_para
'''
from utils.utils_wgan import *

#这里写一个arg参数
if __name__ != "__main__": #
    parser = argparse.ArgumentParser(description="wgan_v1 Argument Parser")
    # Add arguments here
    parser.add_argument("--name",default="ode_neural_operator",type=str, help="ode")
    parser.add_argument("--generator_num",default=1, type=int, help="generator_num ")
    parser.add_argument("--discriminator_num",default=1, type=int, help="discriminator_num ")
    parser.add_argument("--batch_size",default=1, type=int, help="batch_size ")
    parser.add_argument("--num_epochs",default=100, type=int, help="num_epochs ")
    parser.add_argument("--noise_dim",default=1, type=int, help="gaussian_noise_dim ")
    parser.add_argument("--mean",default=0.0, type=float, help="gaussian_mean ")
    parser.add_argument("--stddev",default=0.01, type=float, help="gaussian_stddev ")
    parser.add_argument("--gen_neural_network_deep",default=2, type=int, help="generator_deep ")
    parser.add_argument("--discri_neural_network_deep", default=2, type=int,
                        help="discriminator_deep ")
    parser.add_argument("--seed", default=42, type=int, help="seed ")
    parser.add_argument("--g_neural_network_width", default=512, type=int,
                        help="gen_num_neurons ")
    parser.add_argument("--dis_neural_network_width", default=256, type=int,
                        help="dis_num_neurons ")
    parser.add_argument("--energy_penalty_argu", default=0, type=float,
                        help="energy_penalty_argu-beta")
    parser.add_argument("--argue_basis", default=[1,1], type=list, help="[basis_number"\
                        ",basis_type(1:x^*,2:cos(x))]")
    parser.add_argument("--g_learning_rate", default=1e-3, type=float, help="g_learning_rate")
    parser.add_argument("--d_learning_rate", default=1e-3, type=float, help="d_learning_rate")
    parser.add_argument("--lipschitz_clip", default=0.01, type=float,help="lipschitz_clip")
    parser.add_argument("--iter_generator", default=1, type=int, help="iter_generator")
    parser.add_argument("--activation_function",default=["Relu"],type=list,help="activation")
    parser.add_argument("--savepath",default="../tb_info/compare_multi_argus",type=str,help="model_savepath")
    parser.add_argument("--denote", default="..denote", type=str, help="denote")
    args = parser.parse_args()

    print(args)

