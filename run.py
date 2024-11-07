import sys
import logging
import warnings
import torch
import torchaudio
import librosa
import numpy as np
from models import Detector
from config import *
import os

# 设置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_paths():
    """检查必要的文件和目录是否存在"""
    model_path = 'detector_model'
    audio_path = './watermarked.wav'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"检测器模型目录不存在: {model_path}")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"水印音频文件不存在: {audio_path}")

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

        stft = torch.stft(audio,
                          n_fft=WINDOW_LENGTH,
                          hop_length=HOP_LENGTH,
                          win_length=WINDOW_LENGTH,
                          window=torch.hann_window(WINDOW_LENGTH).to(audio.device),
                          return_complex=True)

        # 分离实部和虚部
        real = stft.real
        imag = stft.imag

        # 将实部和虚部拼接在一起
        stft_combined = torch.stack([real, imag], dim=-1)

        # 调整维度顺序
        stft_data = stft_combined.permute(0, 3, 1, 2)

        logger.info(f"STFT数据形状: {stft_data.shape}")
        return stft_data
    except Exception as e:
        logger.error(f"预处理音频时出错: {str(e)}")
        raise


def main():
    try:
        logger.info("开始检测水印...")

        # 检查CUDA是否可用
        logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
        logger.info(f"当前设备: {DEVICE}")

        # 检查路径
        check_paths()

        # 加载检测器模型
        logger.info("正在加载检测器模型...")
        detector = Detector()
        detector = detector.to(DEVICE)
        detector.eval()
        logger.info("检测器模型加载成功")

        # 加载水印音频
        logger.info("正在加载水印音频...")
        audio, sr = librosa.load('./watermarked.wav', sr=16000)
        logger.info(f"水印音频加载成功, 形状: {audio.shape}, 采样率: {sr}")

        # 预处理音频
        audio_tensor = preprocess_audio(audio).to(DEVICE)

        # 检测水印
        logger.info("正在检测水印...")
        with torch.no_grad():
            detected_watermark = detector(audio_tensor)

        # 将检测到的水印与MESSAGE_POOL比对
        MESSAGE_POOL_tensor = torch.from_numpy(MESSAGE_POOL).to(DEVICE)
        mse_scores = torch.mean((MESSAGE_POOL_tensor - detected_watermark) ** 2, dim=1)
        detected_index = torch.argmin(mse_scores).item()
        min_mse = mse_scores[detected_index].item()

        logger.info(f"检测到的水印索引: {detected_index}")
        logger.info(f"最小均方误差: {min_mse:.6f}")

        # 保存检测结果
        detected_message = MESSAGE_POOL[detected_index]
        np.save('detected_watermark.npy', detected_message)
        logger.info("已保存检测到的水印信息")

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