
import numpy as np
import matplotlib.pyplot as plt
def help_fft(x):
    '''
    输入时间序列,
    输出是频谱 注意：omega=2pi*f
    '''
    # 进行傅里叶变换
    time_series=x
    fft_result = np.fft.fft(time_series)
    # 计算频率轴
    sampling_rate = 50  # 假设采样率为1000Hz 2s/100=0.02
    frequencies = np.fft.fftfreq(len(time_series), 1/sampling_rate)

    # 仅保留单边的频谱信息
    num_samples = len(time_series)
    num_freq = num_samples // 2
    frequencies = frequencies[:num_freq]
    spectrum = np.abs(fft_result)[:num_freq]

    # 进行最大值归一化
    max_value = np.max(spectrum)
    normalized_spectrum = spectrum / max_value


    return frequencies,normalized_spectrum




if __name__ == '__main__':
    a=help_fft()
    plt.show(a)
    print("this is fft")