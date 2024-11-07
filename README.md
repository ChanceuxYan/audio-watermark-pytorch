# Audio Watermarking with Deep Neural Networks (PyTorch Version)

这是一个基于深度神经网络的音频水印系统，使用 PyTorch 实现。该系统能够将数字水印嵌入到音频信号中，并在经过各种信号处理攻击后仍能可靠地检测和提取水印。

## 功能特点

- 基于深度学习的音频水印嵌入和检测
- 支持多种信号处理攻击的鲁棒性测试
- 高质量的音频重建，确保水印不影响音频质量
- 支持批量处理和 GPU 加速

## 系统要求

- Python 3.6+
- PyTorch 1.8+
- CUDA (可选，用于 GPU 加速)

## 安装依赖
torch>=1.8.0
numpy>=1.19.2
tqdm>=4.60.0
soundfile>=0.10.3
tensorflow==2.4.1
librosa==0.7.2
pypesq==1.2.4
