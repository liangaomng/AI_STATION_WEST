import numpy as np
import matplotlib.pyplot as plt
import pywt

wavelet_name = 'db1'
wavelet = pywt.Wavelet(wavelet_name)
level = 4

# 创建绘图
fig, axs = plt.subplots(level, 2, figsize=(12, 8))

wavelet = 'haar'

fig, axarr = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))

for level in range(1, 4):
    wavelet_function = pywt.Wavelet(wavelet)
    phi, psi, x = wavelet_function.wavefun(level=level)

    axarr[level - 1, 0].plot(x, phi)
    axarr[level - 1, 0].set_title(f'Scaling Function (Level {level})')

    axarr[level - 1, 1].plot(x, psi)
    axarr[level - 1, 1].set_title(f'Wavelet Function (Level {level})')

plt.tight_layout()
plt.show()



