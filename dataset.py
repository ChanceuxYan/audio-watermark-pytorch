import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from random import randint
from config import *

def create_message_pool(pool_size=1<<14, message_size=64):
    """
    创建水印消息池
    Args:
        pool_size: 消息池大小
        message_size: 每个消息的大小
    Returns:
        np.array: 消息池
    """
    pool = set()
    while len(pool) < pool_size:
        pool.add(tuple(np.random.randint(0, 2, message_size)))
    return np.array(list(pool))

def generate_random_message(message_pool=MESSAGE_POOL, batch_size=BATCH_SIZE, num_bits=NUM_BITS):
    """
    从消息池中生成随机消息
    """
    ind = randint(0, len(message_pool)-1)
    message = np.broadcast_to(message_pool[ind], (batch_size, num_bits))
    return torch.from_numpy(message).float().to(DEVICE)

def expand_message(message, batch_size=BATCH_SIZE, num_bits=NUM_BITS):
    """
    扩展消息维度以匹配模型要求
    """
    temp = torch.empty((batch_size, 16, 2, num_bits), device=message.device)
    message = message.view(batch_size, 1, 1, num_bits)
    temp[:, :, :, :] = message
    return temp

class AudioDataset(Dataset):
    """
    音频数据集类
    """
    def __init__(self, stft_data):
        """
        Args:
            stft_data: STFT数据
        """
        self.stft_data = torch.from_numpy(stft_data).float()
        
    def __len__(self):
        return len(self.stft_data)
    
    def __getitem__(self, idx):
        stft = self.stft_data[idx]
        message = generate_random_message(batch_size=1)
        expanded_message = expand_message(message, batch_size=1)
        return stft.to(DEVICE), expanded_message.to(DEVICE)

def get_dataloader(stft_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4):
    """
    创建数据加载器
    Args:
        stft_data: STFT数据
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
    Returns:
        DataLoader: PyTorch数据加载器
    """
    dataset = AudioDataset(stft_data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    ) 