import torch
import torch.nn.functional as F
import random
from config import *
import logging

logger = logging.getLogger(__name__)

def random_message():
    """
    生成随机水印消息
    Returns:
        torch.Tensor: 随机选择的水印消息
    """
    try:
        message = random.choice(MESSAGE_POOL)
        return torch.from_numpy(message).float().unsqueeze(0).to(DEVICE)
    except Exception as e:
        logger.error(f"生成随机消息时出错: {e}")
        raise

def embed(embedder, signal, message=None):
    """
    在音频信号中嵌入水印
    Args:
        embedder: 水印嵌入模型
        signal: 输入音频信号 [batch_size, samples]
        message: 要嵌入的水印消息 (可选)
    Returns:
        torch.Tensor: 嵌入水印后的音频信号
    """
    try:
        if message is None:
            message = random_message()
            
        # 确保信号在正确的设备上
        signal = signal.to(DEVICE)
        
        # 计算STFT
        stft = torch.stft(
            signal,
            n_fft=WINDOW_LENGTH,
            hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH,
            window=torch.hann_window(WINDOW_LENGTH).to(DEVICE),
            return_complex=True
        )
        
        # 调整维度并分离实部虚部
        stft = stft.permute(0, 2, 1)
        stft_input = torch.stack([stft.real, stft.imag], dim=-1)
        
        # 嵌入水印
        output = embedder([stft_input, message])
        stft_complex = torch.complex(output[..., 0], output[..., 1])
        
        # 计算ISTFT
        return torch.istft(
            stft_complex.permute(0, 2, 1),
            n_fft=WINDOW_LENGTH,
            hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH,
            window=torch.hann_window(WINDOW_LENGTH).to(DEVICE)
        )
        
    except Exception as e:
        logger.error(f"嵌入水印时出错: {e}")
        raise

def detect(detector, signal):
    """
    从音频信号中检测水印
    Args:
        detector: 水印检测模型
        signal: 输入音频信号 [batch_size, samples]
    Returns:
        torch.Tensor: 检测到的水印消息
    """
    try:
        # 确保信号在正确的设备上
        signal = signal.to(DEVICE)
        
        # 计算STFT
        stft = torch.stft(
            signal,
            n_fft=WINDOW_LENGTH,
            hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH,
            window=torch.hann_window(WINDOW_LENGTH).to(DEVICE),
            return_complex=True
        )
        
        # 调整维度并分离实部虚部
        stft = stft.permute(0, 2, 1)
        stft_input = torch.stack([stft.real, stft.imag], dim=-1)
        
        # 检测水印
        return detector(stft_input)
        
    except Exception as e:
        logger.error(f"检测水印时出错: {e}")
        raise 