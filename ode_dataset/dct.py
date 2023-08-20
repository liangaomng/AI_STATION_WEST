import numpy as np
import matplotlib.pyplot as plt
import pywt
import matplotlib.gridspec as gridspec

# 生成更为复杂的信号
t = np.linspace(0, 2, 2 * 50, endpoint=False)
y = np.sin(2 * t)
y += 0.5 * np.cos(0.1* t)  # 添加高频部分
y += 0.2 * np.random.randn(t.size)  # 添加随机噪声
y[t > 1.4] = 0  # 从 t=1.4 开始的信号部分突然变为0

# 进行DWT
wavelet = 'db1'
coeffs = pywt.wavedec(y, wavelet, level=1)
labels = ['cA4', 'cD4', 'cD3', 'cD2', 'cD1']

# 使用GridSpec创建子图布局
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(len(coeffs), 3, width_ratios=[1, 1, 0.4])

# 获取小波母函数形状
wavelet_function = pywt.Wavelet(wavelet)
phi, psi, x = wavelet_function.wavefun(level=1)

# 对于每一级的小波系数
for i, (coeff, label) in enumerate(zip(coeffs, labels)):
    ti = np.linspace(0, 2, len(coeff), endpoint=False)

    # 绘制小波系数
    ax0 = fig.add_subplot(gs[i, 0])
    ax0.plot(ti, coeff, label=label, color='blue')
    ax0.legend(loc='upper right')

    # 绘制小波母函数或尺度函数的形状
    ax1 = fig.add_subplot(gs[i, 1])
    if 'cA' in label:
        ax1.plot(x * (2 / len(coeff)) + ti[0], phi, label='Scaling function (phi)', color='red')
    else:
        ax1.plot(x * (2 / len(coeff)) + ti[0], psi, label='Wavelet function (psi)', color='green')
    ax1.legend(loc='upper right')

# 在左侧列的中间设置"Value"作为ylabel
fig.text(0.05, 0.5, 'Value', ha='center', va='center', rotation='vertical')

# 用系数重构信号
reconstructed = pywt.waverec(coeffs, wavelet)

# 绘制原始和重构的信号
ax2 = fig.add_subplot(gs[:, 2])
ax2.plot(t, y, color='grey', label='Original Signal')
ax2.plot(t, reconstructed, color='orange', linestyle='--', label='Reconstructed Signal')
ax2.plot(t, reconstructed-y, color='blue', linestyle='--', label='Reconstructed Signal')
print(reconstructed-y)
ax2.legend(loc='upper right')
ax2.set_title('Original & Reconstructed Signals')
ax2.set_xlabel('Time (s)')

fig.suptitle('DWT Coefficients, Wavelet & Scaling Functions, and Original & Reconstructed Signals')
plt.tight_layout()
plt.subplots_adjust(top=0.92, wspace=0.3, left=0.12)
plt.show()
