import numpy as np
import matplotlib.pyplot as plt
import pywt
import matplotlib.gridspec as gridspec



def help_dwt(time_range,
             time_series:list,
             wavelet_str='haar',
             level_num=4,
             solutions_dict=0):
    '''
    for DWT
    input time_series(two functions)
    output dwt-figure
    solutions eg
    {'z1_solu': 'z1=sin(t) + cos(t)', 'z2_solu': 'z2=-sin(t) + cos(t)'}
    '''

    print('solutions:',solutions_dict)

    time=time_range
    #find the max
    t_max=time.max()
    y= time_series
    wavelet = wavelet_str
    coeffs = pywt.wavedec(y, wavelet, level=level_num)
    labels = ['cA4', 'cD4', 'cD3', 'cD2', 'cD1']
    color_list=['k','blue','red']
    # grid
    fig = plt.figure(figsize=(18, 10))
    plt.style.use('ggplot')
    gs = gridspec.GridSpec(len(coeffs), 4, width_ratios=[1, 1, 1,1])

    # mother function shape
    wavelet_function = pywt.Wavelet(wavelet)

    # plot
    for i, (coeff, label) in enumerate(zip(coeffs, labels)):

        phi, psi, x = wavelet_function.wavefun(level=i+1)
        x_coords = list(range(len(coeff[0])))
        # coeffs

        ax0 = fig.add_subplot(gs[i, 0])
        ax0.scatter(x_coords,coeff[0], label='z1(t)'+solutions_dict['z1_solu'],
                    color='blue')
        ax0.scatter(x_coords,coeff[1], label='z2(t)'+solutions_dict['z1_solu'],
                    color='k')
        ax0.legend(loc='upper right')
        ax0.set_title('wavelet coefficient of '+wavelet_str+'_'+labels[i],
                      fontsize=15)
        ax0.set_xlabel('number of coeffs',fontsize=15)

        # plot Wavelet function
        ax1 = fig.add_subplot(gs[i, 1])
        ax1.plot(x , psi,
                     label='Wavelet function (psi)', color='green')
        ax1.set_title('wavelet function of '+wavelet_str,fontsize=15)
        ax1.legend(loc='upper right')
        ax1.set_xlabel('t',fontsize=15)

        # plot Scaling function
        ax2 = fig.add_subplot(gs[i, 2])
        ax2.plot(x , phi,
                     label='Scaling function (phi)', color='red')
        ax2.set_title('scaling function of '+wavelet_str,fontsize=15)
        ax2.legend(loc='upper right')
        ax2.set_xlabel('t',fontsize=15)


    reconstructed = pywt.waverec(coeffs, wavelet)

    fig.text(0.01, 0.5, 'Value', ha='center', va='center',
             rotation='vertical',fontsize=15)

    ax3 = fig.add_subplot(gs[:, 3])
    for i in range(2):
        value = np.sum((reconstructed[i] - y[i])**2)
        rounded_value = round(value, 2)
        formatted_value = "{:.2e}".format(rounded_value)
        color=color_list[i%3]

        ax3.plot(time, y[i], color=color, linestyle='-.',
                 label='Original Signal='+"z"+str(i+1),alpha=0.5,linewidth=3)
        ax3.plot(time, reconstructed[i], color=color, linestyle='--',
                 label='Reconstructed Signal='+"z"+str(i+1),alpha=0.5,linewidth=3)
        ax3.plot(time, reconstructed[i]-y[i], color=color, linestyle=':',
                 label='mse error='+
                 formatted_value,alpha=0.8,linewidth=4)
    ax3.legend(loc='upper left')
    ax3.set_title('Original & Reconstructed',fontsize=15)
    ax3.set_xlabel('Time (s)')

    fig.suptitle('DWT Coefficients, Wavelet & Scaling Functions'\
                 'and Original & Reconstructed Signals',fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, wspace=0.1)



if __name__ == '__main__':
    print("this is for dwt")
    # help_dwt(time_range=np.linspace(0,2,100),time_series=[np.linspace(0, 2, 2*50, endpoint=False),
    #                       np.sin(np.linspace(0, 2, 2*50, endpoint=False))])
    #print(pywt.wavelist(kind='discrete'))
