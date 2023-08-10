import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
from fft import *
from cwt import *
from dwt import *
plt.style.use('ggplot')

def plot_visual(data,solutions:list,savepath,save_number:int):
    ''' 
    data is a tensor style [100,10]
    this is to generate
    1.phase portrait domain
    2.freq domain
    3.time domain
    4.dwt figure
    :return:png
    '''
    # data
    numbers=data.shape [0]
    # phase portrait
    z1_t=data[:,7]
    z2_t=data[:,8]
    time_range=data[:,6]
    ic_z1=data[0,0]
    ic_z2=data[0,1]
    #the matrix
    matrix=data[0,3:7]
    # the title
    title_size=80
    line_size=5
    # figs
    fig, axes = plt.subplots(1, 3, figsize=(100, 30))

    # phase portrait
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-5, 5)
    # axis
    axes[0].plot([0, 0], [-5, 5], color='k', linestyle='--',
                 linewidth=line_size)
    axes[0].plot([-5, 5], [0, 0], color='k', linestyle='--',
                 linewidth=line_size)

    axes[0].plot(z1_t[:], z2_t[:],
                 label=f'Initial Condition {+1}')
    difference_z1 = [x - y for x, y in zip(z1_t[1:], z1_t[:-1])]
    difference_z2 = [x - y for x, y in zip(z2_t[1:], z2_t[:-1])]
    axes[0].quiver(z1_t[:-1], z2_t[:-1],
               difference_z1,
               difference_z2,
               scale_units='xy', angles='xy', scale=0.1,color='r')
    axes[0].set_title("phase portrait",fontsize=title_size)
    axes[0].set_xlabel("z1(t)",fontsize=title_size)
    axes[0].set_ylabel("z2(t)",fontsize=title_size)

    #time domain
    axes[1].set_xlim(0, 2)
    # t and value

    axes[1].plot(time_range, z1_t[:], linewidth=10,linestyle= '--',
                 label=f'Initial Condition { 1}')
    axes[1].plot(time_range, z2_t[:], linewidth=10,linestyle='-.',
                 label=f'Initial Condition { 2}')

    axes[1].set_xlabel('t', fontsize=title_size)
    axes[1].set_ylabel('z', fontsize=title_size)
    axes[1].set_title('time space', fontsize=title_size)
    axes[1].grid(True)
    axes[1].legend(loc='upper right',ncol=10,fontsize=title_size/2)

    # freq domain

    z1_freqs, z1_norm_spec = help_fft(z1_t[:])
    z2_freqs, z2_norm_spec = help_fft(z2_t[:])

    print("z1 the highest freq"+solutions['z1_solu'],
              z1_freqs[np.argmax(z1_norm_spec)])
    print("z2 the highest freq"+solutions['z2_solu'],
              z2_freqs[np.argmax(z2_norm_spec)])

    axes[2].scatter(z1_freqs, z1_norm_spec,s=3000,alpha=0.8,marker='*',
                        label=f'Initial Condition { 1}')
    axes[2].scatter(z2_freqs, z2_norm_spec,s=1000,alpha=0.8,
                        label=f'Initial Condition { 2}')
    #dash line
    for xi in z1_freqs:
        axes[2].axvline(x=xi, linestyle='--',
                            color='blue', alpha=0.5)

    axes[2].set_xlabel('Frequency (Hz)', fontsize=title_size)
    axes[2].set_ylabel('Normalized Amplitude', fontsize=title_size)
    axes[2].set_title('Normalized Frequency Spectrum',
                      fontsize=title_size)
    axes[2].legend(loc='upper right', ncol=3,
                   fontsize=title_size/2)
    axes[2].grid(True)

    #adjust
    for ax in axes:
        # tick size
        ax.tick_params(axis='both', which='major', labelsize=title_size)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(line_size)  # 设置线宽为2

    z1_t=z1_t.reshape(-1)
    z2_t=z2_t.reshape(-1)

    plt.tight_layout()  #
    plt.savefig(savepath+'/'+str(save_number)+'ptf.png')
    # close and save
    plt.close('all')
    help_dwt(time_range=time_range,time_series=[z1_t,z2_t],
             wavelet_str='db1',level_num=4,solutions_dict=solutions)
    plt.savefig(savepath+'/'+str(save_number)+'dwt.png')
    plt.close()
    print("save the figure successfully")
