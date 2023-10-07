import torch
import utlis_2nd.neural_base_class as nn_base

def load_model_eval():
    path="/liangaoming/conda_lam/expriments/paper1/expr1/expr1_98_data/model_check_point/omega_net/omega_net_model.pth"

    S_Omega = nn_base.Omgea_MLPwith_residual_dict(input_dim=400,
                                                  hidden_dims=[512, 512, 512],
                                                  output_dim=2 *51,
                                                  hidden_act="rational",
                                                  output_act="softmax",
                                                  )
    checkpoint=torch.load(path)

    S_Omega.load_state_dict(checkpoint['S_Omega_model_state_dict'])
    S_Omega.eval()
    return S_Omega

import matplotlib.pyplot as plt
if __name__=="__main__":
    print("hi")
    t=torch.linspace(0,2,100)
    x=torch.sin(t)
    y=torch.cos(t)
    tensor=torch.stack([x,y],dim=1)
    tensor=tensor.unsqueeze(0)
    batch,length,vari_number=tensor.shape


    test_omega=nn_base.Omgea_MLPwith_residual_dict(input_sample_lenth=length,
                                                   hidden_dims=[512, 512, 512],
                                                   output_dim=1,
                                                   hidden_act="rational",
                                                   output_act="softmax",
                                                   sample_vesting=2,
                                                  )
    input_tensor=test_omega.convert_data_2_cat_grad(tensor)
    result=test_omega(input_tensor)
    fft_result=test_omega.return_fft_spectrum(input_tensor,need_norm=True,vari_order=1)
    print("fft_result",fft_result.shape)
    print("result",result.shape)
    print("entropy",test_omega.calculate_entropy(result))

    plt.plot(fft_result.squeeze(0).detach().numpy())
    plt.show()


