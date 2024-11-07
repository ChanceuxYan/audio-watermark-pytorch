import torch
from utils import bwh
import numpy as np
import os

# 基础配置
BATCH_SIZE = 64
FS = 16000  # 采样率
FC = 4000   # 截止频率

# 信号和消息形状
SIGNAL_SHAPE = (512, 64, 2)  # 信号的STFT表示形状
MESSAGE_SHAPE = (16, 2, 512) # 水印消息形状
NUM_BITS = 512               # 水印比特数

# 攻击参数
COEFFICIENT = 15             # 攻击概率系数
NOISE_STRENGTH = 0.009      
NUM_SAMPLES = 1000          # 采样数量

# STFT参数
HOP_LENGTH = 504            # 帧移
STEP_SIZE = 32768 - HOP_LENGTH
WINDOW_LENGTH = 1023        # 窗口长度

# 水印池配置
POOL_SIZE = 6
MESSAGE_POOL_PATH = "dataset/message_pool.npy"
if os.path.exists(MESSAGE_POOL_PATH):
    MESSAGE_POOL = np.load(MESSAGE_POOL_PATH)
else:
    raise FileNotFoundError(f"未找到message_pool.npy文件: {MESSAGE_POOL_PATH}")

# Butterworth滤波器配置
BUTTERWORTH = bwh(n=16, fc=FC, fs=FS, length=25)

# PyTorch特定配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100

# 模型保存路径
MODEL_SAVE_DIR = "checkpoints"
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR) 