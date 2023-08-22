
import numpy as np
import matplotlib.pyplot as plt
def help_fft(x):
    '''
    input: x is a list
            omega=2pi*f
    output:f
    '''
    # sub mean value and fft
    time_series=x
    time_series=time_series-np.mean(time_series)
    fft_result = np.fft.fft(time_series)
    # fft
    sampling_rate = 50  #  2s/100=0.02
    frequencies = np.fft.fftfreq(len(time_series), 1/sampling_rate)

    # single side spectrum
    num_samples = len(time_series)
    num_freq = num_samples // 2
    frequencies = frequencies[:num_freq]
    spectrum = np.abs(fft_result)[:num_freq]

    # max norm
    max_value = np.max(spectrum)
    normalized_spectrum = spectrum / max_value


    return frequencies,normalized_spectrum




if __name__ == '__main__':
    a=help_fft()
    plt.show(a)
    print("this is fft")