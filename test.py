import sys
import logging
import warnings

warnings.filterwarnings("ignore", message="Key already registered with the same priority")

# 设置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import torch
import torchaudio
import librosa
import numpy as np
from models import Embedder, prepare_message
from config import *
import os



def check_paths():
    """检查必要的文件和目录是否存在"""
    model_path = 'embedder_model'
    audio_path = './samples/example1.wav'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型目录不存在: {model_path}")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    logger.info("所有必要的文件路径检查通过")


def preprocess_audio(audio, sr=16000):
    """预处理音频数据"""
    try:
        logger.info("开始预处理音频...")
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        logger.info(f"音频张量形状: {audio.shape}")

        # window = torch.hann_window(512).to(audio.device)
        stft = torch.stft(audio,
                          n_fft=WINDOW_LENGTH,
                          hop_length= HOP_LENGTH,
                          win_length=WINDOW_LENGTH,
                          window=torch.hann_window(WINDOW_LENGTH).to(audio.device),
                          return_complex=True)  # 改为 True，返回复数张量

        # 分离实部和虚部
        real = stft.real
        imag = stft.imag

        # 将实部和虚部拼接在一起
        stft_combined = torch.stack([real, imag], dim=-1)

        # 调整维度顺序以匹配 TensorFlow 的输出
        stft_data = stft_combined.permute(0, 3, 1, 2)

        logger.info(f"STFT数据形状: {stft_data.shape}")
        return stft_data
    except Exception as e:
        logger.error(f"预处理音频时出错: {str(e)}")
        raise


def postprocess_audio(stft_data):
    """
    将 STFT 数据转换回音频信号
    Args:
        stft_data: 形状为 [batch_size, 2, 512, 64] 的张量，第二个维度是实部和虚部
    Returns:
        重建的音频信号
    """
    try:
        logger.info("开始后处理音频...")
        logger.info(f"输入STFT数据形状: {stft_data.shape}")

        # 调整维度顺序，从 [batch_size, 2, freq, time] 到 [batch_size, freq, time, 2]
        stft_data = stft_data.permute(0, 2, 3, 1)

        # 将实部和虚部转换为复数张量
        stft_complex = torch.complex(stft_data[..., 0], stft_data[..., 1])

        # 执行 ISTFT
        audio = torch.istft(
            stft_complex,
            n_fft=WINDOW_LENGTH,
            hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH,
            window=torch.hann_window(WINDOW_LENGTH).to(stft_complex.device),
            return_complex=False
        )

        # 如果输入是批处理的，取第一个样本
        if len(audio.shape) > 1:
            audio = audio[0]

        # 转换为 numpy 数组
        audio = audio.cpu().detach().numpy()

        # 确保音频值在 [-1, 1] 范围内
        audio = np.clip(audio, -1, 1)

        logger.info(f"重建音频形状: {audio.shape}")
        return audio
    except Exception as e:
        logger.error(f"后处理音频时发生错误: {e}")
        raise e


    except Exception as e:
        logger.error(f"后处理音频时出错: {str(e)}")
        raise


def main():
    try:
        logger.info("程序开始执行...")

        # 检查CUDA是否可用
        logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
        logger.info(f"当前设备: {DEVICE}")

        # 检查路径
        check_paths()

        # 打印Python和PyTorch版本
        logger.info(f"Python版本: {sys.version}")
        logger.info(f"PyTorch版本: {torch.__version__}")

        # # 加载预训练的嵌入器模型
        # logger.info("正在加载模型...")
        # embedder = Embedder.from_tensorflow('embedder_model')
        # embedder = embedder.to(DEVICE)
        # embedder.eval()
        # logger.info("模型加载成功")
        embedder = Embedder()

        # 加载音频
        logger.info("正在加载音频...")
        audio, sr = librosa.load('./samples/example1.wav', sr=16000)
        logger.info(f"音频加载成功, 形状: {audio.shape}, 采样率: {sr}")

        # 预处理音频
        audio_tensor = preprocess_audio(audio).to(DEVICE)

        # 创建水印消息
        logger.info("正在生成水印消息...")
        message = prepare_message()
        # message = torch.randint(0, 2, (1, 512), device=DEVICE).float()
        # message_index = 3
        # # 将 MESSAGE_POOL 转换为 PyTorch tensor
        # message = torch.from_numpy(MESSAGE_POOL[message_index])
        # # 添加维度并复制到指定形状 (1, 16, 2, 512)
        # message = message.unsqueeze(0).unsqueeze(0)  # 添加两个维度变成 (1, 1, 512)
        # message = message.repeat(1, 16, 2, 1)  # 扩展到 (1, 16, 2, 512)

        # 嵌入水印
        logger.info("正在嵌入水印...")
        with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            embedder = embedder.to(device)  # 将模型移动到 GPU（如果可用）
            audio_tensor = audio_tensor.to(device)  # 确保输入也在 GPU 上

            watermarked_stft = embedder(audio_tensor, message)
            print(watermarked_stft.shape)

        # 后处理并保存结果
        logger.info("正在保存水印音频...")
        watermarked_audio = postprocess_audio(watermarked_stft)
        import soundfile as sf
        sf.write('watermarked.wav', watermarked_audio, sr)
        logger.info("已成功保存水印音频到 watermarked.wav")

    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序异常退出: {str(e)}")
        sys.exit(1)