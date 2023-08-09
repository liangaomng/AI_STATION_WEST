import numpy as np
import matplotlib.pyplot as plt
import pywt
import matplotlib.gridspec as gridspec



def help_dwt(time_series,wavelet_str='db1',level_num=4):
    '''
     for DWT
     '''
    t= np.linspace(0, 2, 2*50, endpoint=False)
    y= time_series
    wavelet = wavelet_str
    coeffs = pywt.wavedec(y, wavelet, level=level_num)
    labels = ['cA4', 'cD4', 'cD3', 'cD2', 'cD1']

    # 使用GridSpec创建子图布局
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(len(coeffs), 3, width_ratios=[1, 1, 0.4])

    # 获取小波母函数形状
    wavelet_function = pywt.Wavelet(wavelet)
    phi, psi, x = wavelet_function.wavefun(level=level_num)

    # 对于每一级的小波系数
    for i, (coeff, label) in enumerate(zip(coeffs, labels)):
        ti = np.linspace(0, 2, len(coeff[0]), endpoint=False)

        # 绘制小波系数
        ax0 = fig.add_subplot(gs[i, 0])

        ax0.scatter(ti ,coeff[0], label=label, color='blue')
        ax0.legend(loc='upper right')
        ax0.set_title('wavelet coefficient of '+wavelet_str,fontsize=15)
        ax0.set_xlabel('t',fontsize=15)

        # 绘制小波母函数或尺度函数的形状
        ax1 = fig.add_subplot(gs[i, 1])
        if 'cA' in label:
            ax1.plot(x * (2 / len(coeff)) + ti[0], phi,
                     label='Scaling function (phi)', color='red')
        else:
            ax1.plot(x * (2 / len(coeff)) + ti[0], psi,
                     label='Wavelet function (psi)=', color='green')
        ax1.set_title('wavelet function of '+wavelet_str,fontsize=15)
        ax1.legend(loc='upper right')
        ax1.set_xlabel('t',fontsize=15)

    # 用系数重构信号
    reconstructed = pywt.waverec(coeffs, wavelet)
    # 在左侧列的中间设置"Value"作为ylabel
    fig.text(0.01, 0.5, 'Value', ha='center', va='center',
             rotation='vertical',fontsize=15)
    # # 绘制原始和重构的信号
    # ax2 = fig.add_subplot(gs[:, 2])
    # ax2.plot(t, y, color='red', linestyle='-.',label='Original Signal')
    # ax2.plot(t, reconstructed, color='orange', linestyle='--', label='Reconstructed Signal')
    # ax2.plot(t, reconstructed-y, color='green', linestyle=':', label='abs error')
    # ax2.legend(loc='upper right')
    # ax2.set_title('Original & Reconstructed Signals',fontsize=15)
    # ax2.set_xlabel('Time (s)')
    #
    # fig.suptitle('DWT Coefficients, Wavelet & Scaling Functions, and Original & Reconstructed Signals for y=sin(t)')
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.92, wspace=0.1)
    plt.show()

if __name__ == '__main__':
    help_dwt(time_series=[np.linspace(0, 2, 2*50, endpoint=False),
                          np.sin(np.linspace(0, 2, 2*50, endpoint=False))])
    print(pywt.wavelist())
