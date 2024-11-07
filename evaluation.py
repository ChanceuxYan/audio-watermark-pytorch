import torch
import numpy as np
import librosa
from pypesq import pesq
from utils import snr
from config import *
import logging

logger = logging.getLogger(__name__)

def reconstruct_from_stft(example):
    """
    从STFT重建音频信号
    Args:
        example: STFT表示 (shape: [512, 64, 2])
    Returns:
        np.array: 重建的音频信号
    """
    stft = torch.complex(example[..., 0], example[..., 1])
    signal = librosa.core.istft(
        stft.cpu().numpy(),
        hop_length=HOP_LENGTH,
        win_length=WINDOW_LENGTH-1,
        center=True
    )
    return signal

def test_step(model, batch):
    """
    单步测试
    Args:
        model: 水印模型
        batch: (input_stfts, input_messages)
    Returns:
        tuple: (正确比特数, PESQ分数, SNR分数, 有效样本数)
    """
    model.eval()
    with torch.no_grad():
        input_stfts, input_messages = [x.to(DEVICE) for x in batch]
        
        # 获取模型输出
        encoder_output, attacks_output, decoder_output = model(input_stfts, input_messages)
        
        # 二值化检测结果
        output_messages = (decoder_output >= 0.5).float()
        input_messages = input_messages[:, 0, 0, :].long()
        
        # 计算正确掩码
        mask = (output_messages == input_messages).float()
        
        total_pesq = 0
        total_snr = 0
        count = 0
        remove_indices = []
        
        # 计算每个样本的PESQ和SNR
        for i in range(len(batch[0])):
            input_signal = reconstruct_from_stft(input_stfts[i].cpu())
            output_signal = reconstruct_from_stft(encoder_output[i].cpu())
            
            curr_pesq = pesq(input_signal, output_signal, FS)
            if not np.isnan(curr_pesq):
                total_pesq += curr_pesq
                total_snr += snr(input_signal, output_signal)
                count += 1
            else:
                remove_indices.append(i)
        
        if remove_indices:
            mask = np.delete(mask.cpu().numpy(), remove_indices, axis=0)
        
        return mask.sum().item(), total_pesq, total_snr, count

def test(model, test_loader, verbose=False):
    """
    测试模型
    Args:
        model: 水印模型
        test_loader: 测试数据加载器
        verbose: 是否打印详细信息
    Returns:
        tuple: (平均准确率, 平均PESQ分数, 平均SNR分数)
    """
    logger.info("开始测试...")
    model.eval()
    
    total_acc = 0
    total_pesq = 0
    total_snr = 0
    count = 0
    step = 1
    
    with torch.no_grad():
        for batch in test_loader:
            batch_acc, batch_pesq, batch_snr, batch_count = test_step(model, batch)
            
            if verbose:
                logger.info(
                    f"Batch {step} - "
                    f"Accuracy: {batch_acc/(batch_count*NUM_BITS):.4f}, "
                    f"PESQ: {batch_pesq/batch_count:.4f}, "
                    f"SNR: {batch_snr/batch_count:.4f}"
                )
            
            total_acc += batch_acc
            total_pesq += batch_pesq
            total_snr += batch_snr
            count += batch_count
            step += 1
    
    return total_acc/(count*NUM_BITS), total_pesq/count, total_snr/count

def load_model(model, checkpoint_path):
    """
    加载模型检查点
    Args:
        model: 水印模型
        checkpoint_path: 检查点路径
    Returns:
        model: 加载检查点后的模型
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"成功加载模型检查点: {checkpoint_path}")
        return model
    except Exception as e:
        logger.error(f"加载模型检查点失败: {e}")
        raise 