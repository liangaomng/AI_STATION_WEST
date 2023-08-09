import numpy as np
import matplotlib.pyplot as plt
import pywt

def help_cwt(time_range,time_series,scale_hlimit=1024,wavelet_str='morl'):
    '''
    # cwt
    '''
    signal=time_series
    fs=50
    t=time_range
    coefficients, frequencies = pywt.cwt(signal, scales=np.arange(1, scale_hlimit),
                                         wavelet=wavelet_str, sampling_period=1/fs)

    # 对每一个时刻，找到最大值所对应的频率
    max_freq_per_time = frequencies[np.argmax(np.abs(coefficients), axis=0)]

    # 绘制等高线图
    plt.figure(figsize=(10, 5))
    contour = plt.contour(t, frequencies, np.abs(coefficients), levels=20, cmap='viridis')
    plt.colorbar(label="Magnitude")
    plt.clabel(contour, inline=1, fontsize=10)

    # 在图上绘制每个时刻的最大频率
    plt.plot(t, max_freq_per_time, 'r-', label='Max Frequency per Time')
    plt.scatter(t, max_freq_per_time, color='red', s=10)
    plt.ylim(0,2)

    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Continuous Wavelet Transform (CWT)  '\
              f'Max Frequency per Time and mean freq={np.mean(max_freq_per_time):.2f}',
                fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("mean",np.mean(max_freq_per_time))
if __name__ == '__main__':
    help_cwt(time_range=np.linspace(0, 2, 2*50, endpoint=False),
             time_series=np.sin(np.linspace(0, 2, 2*50, endpoint=False)))
