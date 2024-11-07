import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import randint
from utils import butter_lowpass_filter
from config import *

class LowpassFilter(nn.Module):
    """
    低通滤波攻击层
    """
    def __init__(self, hop_length=HOP_LENGTH, step_size=STEP_SIZE, window_length=WINDOW_LENGTH):
        super(LowpassFilter, self).__init__()
        self.hop_length = hop_length
        self.step_size = step_size
        self.window_length = window_length
        self.window = torch.hann_window(window_length).to(DEVICE)

    def forward(self, inputs):
        print("filtfilt")
        # 转换为复数
        stfts = torch.complex(inputs[..., 0], inputs[..., 1])
        
        # ISTFT
        signals = torch.istft(stfts.permute(0, 2, 1), 
                            self.window_length, 
                            self.hop_length,
                            window=self.window,
                            return_complex=False)
        
        # 应用滤波器
        filtered_signals = []
        for signal in signals:
            signal_np = signal.cpu().numpy()
            filtered = butter_lowpass_filter(signal_np)
            filtered_tensor = torch.tensor(filtered, dtype=torch.float32, device=DEVICE)
            filtered_signals.append(filtered_tensor)
        
        filtered_signals = torch.stack(filtered_signals)
        
        # STFT
        result_stfts = torch.stft(filtered_signals, 
                                self.window_length,
                                self.hop_length,
                                window=self.window,
                                return_complex=True)
        result_stfts = result_stfts.permute(0, 2, 1)
        
        return torch.stack([result_stfts.real, result_stfts.imag], dim=-1)

class AdditiveNoise(nn.Module):
    """
    加性噪声攻击层
    """
    def __init__(self, noise_strength=NOISE_STRENGTH, coefficient=COEFFICIENT,
                 hop_length=HOP_LENGTH, step_size=STEP_SIZE, window_length=WINDOW_LENGTH):
        super(AdditiveNoise, self).__init__()
        self.noise_strength = noise_strength
        self.coefficient = coefficient
        self.hop_length = hop_length
        self.step_size = step_size
        self.window_length = window_length
        self.window = torch.hann_window(window_length).to(DEVICE)

    def forward(self, inputs):
        coefficient = randint(1, 100)
        if coefficient <= self.coefficient:
            print("AdditiveNoise")
            stfts = torch.complex(inputs[..., 0], inputs[..., 1])
            
            signals = torch.istft(stfts.permute(0, 2, 1),
                                self.window_length,
                                self.hop_length,
                                window=self.window,
                                return_complex=False)
            
            noise = torch.rand_like(signals, device=DEVICE) * self.noise_strength
            noise_signals = signals + noise
            
            result_stfts = torch.stft(noise_signals,
                                    self.window_length,
                                    self.hop_length,
                                    window=self.window,
                                    return_complex=True)
            result_stfts = result_stfts.permute(0, 2, 1)
            
            return torch.stack([result_stfts.real, result_stfts.imag], dim=-1)
        return inputs

class CuttingSamples(nn.Module):
    """
    样本切割攻击层
    """
    def __init__(self, num_samples=NUM_SAMPLES, coefficient=COEFFICIENT,
                 batch_size=BATCH_SIZE, input_dim=(33215, 1),
                 hop_length=HOP_LENGTH, step_size=STEP_SIZE, window_length=WINDOW_LENGTH):
        super(CuttingSamples, self).__init__()
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.coefficient = coefficient
        self.hop_length = hop_length
        self.step_size = step_size
        self.window_length = window_length
        self.window = torch.hann_window(window_length).to(DEVICE)

    def forward(self, inputs):
        coefficient = randint(1, 100)
        if coefficient <= self.coefficient:
            print("CuttingSamples")
            stfts = torch.complex(inputs[..., 0], inputs[..., 1])
            
            signals = torch.istft(stfts.permute(0, 2, 1),
                                self.window_length,
                                self.hop_length,
                                window=self.window,
                                return_complex=False)
            
            # 创建切割掩码
            mask = torch.ones(signals.shape, device=DEVICE)
            for i in range(self.batch_size):
                indices = torch.randint(0, signals.shape[1], (self.num_samples,), device=DEVICE)
                mask[i, indices] = 0
            
            signals = signals * mask
            
            result_stfts = torch.stft(signals,
                                    self.window_length,
                                    self.hop_length,
                                    window=self.window,
                                    return_complex=True)
            result_stfts = result_stfts.permute(0, 2, 1)
            
            return torch.stack([result_stfts.real, result_stfts.imag], dim=-1)
        return inputs

class ButterworthFilter(nn.Module):
    """
    Butterworth滤波攻击层
    """
    def __init__(self, butterworth=BUTTERWORTH, coefficient=COEFFICIENT,
                 hop_length=HOP_LENGTH, step_size=STEP_SIZE, window_length=WINDOW_LENGTH):
        super(ButterworthFilter, self).__init__()
        self.butterworth = torch.tensor(butterworth, device=DEVICE).float()
        kernel_size = len(self.butterworth)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding='same', bias=False)
        self.conv.weight.data = self.butterworth.view(1, 1, -1)
        self.conv.weight.requires_grad = False
        self.coefficient = coefficient
        self.hop_length = hop_length
        self.step_size = step_size
        self.window_length = window_length
        self.window = torch.hann_window(window_length).to(DEVICE)

    def forward(self, inputs):
        coefficient = randint(1, 100)
        if coefficient <= self.coefficient:
            print("ButterworthFilter")
            stfts = torch.complex(inputs[..., 0], inputs[..., 1])
            
            signals = torch.istft(stfts.permute(0, 2, 1),
                                self.window_length,
                                self.hop_length,
                                window=self.window,
                                return_complex=False)
            
            signals = signals.unsqueeze(1)  # 添加通道维度
            filtered_signals = self.conv(signals)
            filtered_signals = filtered_signals.squeeze(1)
            
            result_stfts = torch.stft(filtered_signals,
                                    self.window_length,
                                    self.hop_length,
                                    window=self.window,
                                    return_complex=True)
            result_stfts = result_stfts.permute(0, 2, 1)
            
            return torch.stack([result_stfts.real, result_stfts.imag], dim=-1)
        return inputs 