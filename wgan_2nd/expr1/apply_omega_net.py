import torch
import utlis_2nd.neural_base_class as nn_base

def load_model_eval():
    path="/liangaoming/conda_lam/expriments/paper1/expr1/expr1_98_data/model_check_point/omega_net/omega_net_model.pth"

    S_Omega = nn_base.Omgea_MLPwith_residual_dict( input_sample_lenth=length,
                                                   hidden_dims=[512, 512, 512],
                                                   output_coeff=False,
                                                   hidden_act="rational",
                                                   output_act="softmax",
                                                   sample_vesting=2,
                                                   vari_number=2
                                                  )
    checkpoint=torch.load(path)

    S_Omega.load_state_dict(checkpoint['S_Omega_model_state_dict'])
    S_Omega.eval()
    return S_Omega

import matplotlib.pyplot as plt
if __name__=="__main__":
    print("hi")
    t=torch.linspace(0,2,100)

    x=torch.sin(torch.tensor(60)*t)+torch.cos(t)+10+torch.sin(torch.tensor(30)*t)
    y=torch.sin(torch.tensor(60)*t)
    tensor=torch.stack([x,y],dim=1)
    tensor=tensor.unsqueeze(0)
    tensor=tensor.repeat(1,1,1).to("cuda")

    batch,length,vari_number=tensor.shape

    test_Infer=nn_base.Omgea_MLPwith_residual_dict(
                                                   input_sample_lenth=length,
                                                   hidden_dims=[512, 512, 512],
                                                   output_coeff=True,
                                                   hidden_act="rational",
                                                   output_act="Identity",
                                                   sample_vesting=2,
                                                   vari_number=2
                                                  ).to("cuda")

    total_params = sum(p.numel() for p in test_Infer.parameters())
    print(f"Total number of parameters: {total_params}")


    input_tensor=tensor
    result=test_Infer(input_tensor) # [batch,102,2]

    print("result",result.shape)

    coeff_tensor=torch.randn(10,102,2).to("cuda")

    fft_result=test_Infer.return_fft_spectrum(input_tensor,need_norm=True,vari_numbers=2)


    left,pred=test_Infer.return_pred_data(coeff_tensor,fft_result)

    pred_fft=test_Infer.return_fft_spectrum(pred,need_norm=True,vari_numbers=2)
    plt.plot(pred_fft[0,:,0].cpu().detach().numpy())
    plt.show()
    #
    # print("fft_result",fft_result.shape)
    # print("result",result.shape)
    # print("entropy",test_omega.calculate_entropy(fft_result))




