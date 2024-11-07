import torch
import numpy as np
from scipy.signal import butter, filtfilt
from math import pi, sin, cos, sqrt
from cmath import exp

def butter_lowpass(cutoff, sr=16000, order=5):
    """
    设计巴特沃斯低通滤波器
    Args:
        cutoff: 截止频率
        sr: 采样率
        order: 滤波器阶数
    Returns:
        tuple: 滤波器系数 (b, a)
    """
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff=4000, sr=16000, order=16):
    """
    应用巴特沃斯低通滤波器
    Args:
        data: 输入信号
        cutoff: 截止频率
        sr: 采样率
        order: 滤波器阶数
    Returns:
        np.array: 滤波后的信号
    """
    b, a = butter_lowpass(cutoff, sr, order=order)
    return filtfilt(b, a, data)

def bwsk(k, n):
    """
    计算巴特沃斯传递函数的第k个极点
    Args:
        k: 极点索引
        n: 滤波器阶数
    Returns:
        complex: 复数形式的极点
    """
    arg = pi * (2 * k + n - 1) / (2 * n)
    return complex(cos(arg), sin(arg))

def bwj(k, n):
    """
    计算 (s - s_k) * H(s)
    Args:
        k: 极点索引
        n: 滤波器阶数
    Returns:
        complex: 计算结果
    """
    res = complex(1, 0)
    for m in range(1, n + 1):
        if m == k:
            continue
        res /= (bwsk(k, n) - bwsk(m, n))
    return res

def bwh(n=16, fc=400, fs=16e3, length=25):
    """
    计算巴特沃斯滤波器的时域响应
    Args:
        n: 滤波器阶数
        fc: 截止频率
        fs: 采样率
        length: 长度(ms)
    Returns:
        list: 时域响应
    """
    omegaC = 2 * pi * fc
    dt = 1/fs
    number_of_samples = int(fs * length/1000)
    result = []
    
    for x in range(number_of_samples):
        res = complex(0, 0)
        if x >= 0:
            for k in range(1, n + 1):
                res += (exp(omegaC * x * dt/sqrt(2) * bwsk(k, n)) * bwj(k, n))
        result.append(res.real)
    return result

def snr(input_signal, output_signal):
    """
    计算信噪比
    Args:
        input_signal: 输入信号
        output_signal: 输出信号
    Returns:
        float: SNR值(dB)
    """
    if isinstance(input_signal, torch.Tensor):
        input_signal = input_signal.cpu().numpy()
    if isinstance(output_signal, torch.Tensor):
        output_signal = output_signal.cpu().numpy()
        
    Ps = np.sum(np.abs(input_signal ** 2))
    Pn = np.sum(np.abs((input_signal - output_signal) ** 2))
    return 10 * np.log10((Ps/Pn)) 