# 音频水印系统使用教程 (PyTorch 版本)

本教程将指导您如何使用基于 PyTorch 的音频水印系统。该系统支持将数字水印嵌入音频中并能够在经过各种信号处理后检测出水印。

## 1. 项目结构

pytorch_version/
├── config.py          # 配置参数
├── models.py          # 模型定义
├── attacks.py         # 攻击实现
├── dataset.py         # 数据处理
├── train.py          # 训练逻辑
├── evaluation.py      # 评估函数
├── utils.py          # 工具函数
├── main.py           # 主要接口
└── run.py            # 运行示例

## 2. 环境准备

### 2.1 安装依赖
安装必要的包
pip install torch numpy librosa soundfile tensorflow scipy tqdm pypesq

## 3. 基本使用

### 3.1 水印嵌入
python
import torch
from models import Embedder
from config import DEVICE
import librosa
import soundfile as sf
加载预训练的嵌入器模型
embedder = Embedder.from_tensorflow('embedder_model')
embedder = embedder.to(DEVICE)
加载音频
audio, sr = librosa.load('input.wav', sr=16000)
audio_tensor = torch.from_numpy(audio).float().to(DEVICE)
嵌入水印
watermarked = embedder(audio_tensor)
保存结果
sf.write('watermarked.wav', watermarked.cpu().numpy(), sr)
Apply
Copy

### 3.2 水印检测
python
from models import Detector
加载预训练的检测器模型
detector = Detector.from_tensorflow('detector_model')
detector = detector.to(DEVICE)
检测水印
detected_watermark = detector(watermarked)

## 4. 攻击测试
python
from attacks import LowpassFilter, AdditiveNoise, CuttingSamples, ButterworthFilter
创建攻击实例
lowpass = LowpassFilter().to(DEVICE)
noise = AdditiveNoise().to(DEVICE)
cutting = CuttingSamples().to(DEVICE)
butterworth = ButterworthFilter().to(DEVICE)
应用攻击
attacked_audio = lowpass(watermarked)
attacked_audio = noise(attacked_audio)
attacked_audio = cutting(attacked_audio)
attacked_audio = butterworth(attacked_audio)
检测经过攻击的水印
detected_after_attack = detector(attacked_audio)

## 5. 评估
python
from evaluation import test
from dataset import get_dataloader
准备测试数据
test_loader = get_dataloader(your_test_data)
运行评估
accuracy, pesq_score, snr_score = test(
model=(embedder, detector),
test_loader=test_loader,
verbose=True
)
print(f"准确率: {accuracy:.4f}")
print(f"PESQ分数: {pesq_score:.4f}")
print(f"SNR分数: {snr_score:.4f}")

## 6. 注意事项

1. 数据要求：
   - 音频采样率：16kHz
   - 音频格式：WAV推荐
   - 确保音频质量良好

2. 模型使用：
   - 确保模型文件路径正确
   - 注意GPU内存使用
   - 检查输入数据维度

3. 性能优化：
   - 适当调整批处理大小
   - 监控GPU内存使用
   - 必要时使用CPU模式

## 7. 常见问题解决

1. 模型加载错误：
   - 检查模型文件路径
   - 确认TensorFlow模型格式
   - 验证转换过程

2. 内存问题：
   - 减小批处理大小
   - 使用CPU模式
   - 清理GPU缓存

3. 性能问题：
   - 检查GPU使用情况
   - 优化数据加载
   - 调整处理参数

## 8. 参考配置

## 6. 注意事项

1. 数据要求：
   - 音频采样率：16kHz
   - 音频格式：WAV推荐
   - 确保音频质量良好

2. 模型使用：
   - 确保模型文件路径正确
   - 注意GPU内存使用
   - 检查输入数据维度

3. 性能优化：
   - 适当调整批处理大小
   - 监控GPU内存使用
   - 必要时使用CPU模式

## 7. 常见问题解决

1. 模型加载错误：
   - 检查模型文件路径
   - 确认TensorFlow模型格式
   - 验证转换过程

2. 内存问题：
   - 减小批处理大小
   - 使用CPU模式
   - 清理GPU缓存

3. 性能问题：
   - 检查GPU使用情况
   - 优化数据加载
   - 调整处理参数

## 8. 参考配置

## 6. 注意事项

1. 数据要求：
   - 音频采样率：16kHz
   - 音频格式：WAV推荐
   - 确保音频质量良好

2. 模型使用：
   - 确保模型文件路径正确
   - 注意GPU内存使用
   - 检查输入数据维度

3. 性能优化：
   - 适当调整批处理大小
   - 监控GPU内存使用
   - 必要时使用CPU模式

## 7. 常见问题解决

1. 模型加载错误：
   - 检查模型文件路径
   - 确认TensorFlow模型格式
   - 验证转换过程

2. 内存问题：
   - 减小批处理大小
   - 使用CPU模式
   - 清理GPU缓存

3. 性能问题：
   - 检查GPU使用情况
   - 优化数据加载
   - 调整处理参数

## 8. 参考配置
python
config.py 中的关键参数
BATCH_SIZE = 64
FS = 16000 # 采样率
WINDOW_LENGTH = 1023 # STFT窗口长度
HOP_LENGTH = 494 # STFT帧移
NOISE_STRENGTH = 0.009 # 噪声强度

## 9. 使用建议

1. 音频处理：
   - 保持采样率一致
   - 避免多次重采样
   - 注意音频长度

2. 水印强度：
   - 根据需求调整参数
   - 平衡不可闻性和鲁棒性
   - 进行多次测试

3. 攻击测试：
   - 使用多种攻击组合
   - 验证检测准确率
   - 记录测试结果
