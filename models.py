import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from attacks import LowpassFilter, AdditiveNoise, CuttingSamples, ButterworthFilter
import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Embedder(nn.Module):

    """
    水印嵌入器模型 (U-Net架构)
    
    输入:
        - STFT: (batch_size, 2, 512, 64)
        - Message: (batch_size, 16, 2, 512)
    输出:
        - Modified STFT: (batch_size, 2, 512, 64)
    """

    def __init__(self):
        super(Embedder, self).__init__()

        self.bn_7 = nn.BatchNorm2d(256)
        self.bn_8 = nn.BatchNorm2d(128)
        self.bn_9 = nn.BatchNorm2d(64)
        self.bn_10 = nn.BatchNorm2d(32)
        self.bn_11 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # 编码器路径
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.enc6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.enc7 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        # 消息处理
        self.message_conv = nn.Sequential(
            nn.Conv2d(256 + 16, 256, kernel_size=5, padding=2),  # 256是编码器输出通道数
            nn.ReLU()
        )
        
        # 解码器路径
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        )
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        )
        
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(32, 8, kernel_size=5, stride=2, padding=2, output_padding=1)
        )
        
        self.final_conv = nn.Conv2d(16, 2, kernel_size=5, padding=2)

    def forward(self, x, message=None):
        """
        前向传播
        Args:
            x: 输入STFT, 形状为 [batch_size, 2, 512, 64]
            message: 可选，水印消息。如果提供，形状应为 [batch_size, 512] 或 [512]
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if message is not None:
            message = prepare_message()
        else:
            # 使用随机消息
            message = torch.randint(0, 2, (x.size(0), 512), device=x.device).float()
            message = prepare_message()

        # print(x.shape)

        # 编码器路径
        x0 = self.enc1(x)  # [batch_size, 8, 512, 64]
        x1 = self.enc2(x0)  # [batch_size, 16, 256, 32]
        x2 = self.enc3(x1)  # [batch_size, 32, 128, 16]
        x3 = self.enc4(x2)  # [batch_size, 64, 64, 8]
        x4 = self.enc5(x3)  # [batch_size, 128, 32, 4]
        x5 = self.enc6(x4)  # [batch_size, 256, 16, 2]

        # 调整消息的空间维度以匹配x6
        message = message.permute(0, 1, 2, 3)  # [batch_size, 16, 2, 512]
        message = message.permute(0, 3, 1, 2)
        # 调整消息大小以匹配x6的空间维度
        # message = F.interpolate(message, size=(16, 2), mode='bilinear', align_corners=False)

        # 合并特征
        x = torch.cat([x5, message], dim=1)  # [batch_size, 256+16, 16, 2]
        # x = self.message_conv(x)

        # print(message.shape)
        # print(x5.shape)
        # print(x.shape)

        x6 = self.enc7(x)

        # 解码器路径（带跳跃连接）
        un1 = self.dec1(x6)
        un1 = torch.cat([un1, x4], dim=1)

        x7 = self.bn_7(un1)
        x7 = self.relu(x7)
        x7 = self.dropout(x7)

        un2 = self.dec2(x7)
        un2 = torch.cat([un2, x3], dim=1)

        x8 = self.bn_8(un2)
        x8 = self.relu(x8)
        x8 = self.dropout(x8)

        un3 = self.dec3(x8)
        un3 = torch.cat([un3, x2], dim=1)

        x9 = self.bn_9(un3)
        x9 = self.relu(x9)
        x9 = self.dropout(x9)

        un4 = self.dec4(x9)
        un4 = torch.cat([un4, x1], dim=1)

        x10 = self.bn_10(un4)
        x10 = self.relu(x10)

        un5 = self.dec5(x10)
        un5 = torch.cat([un5, x0], dim=1)

        x11 = self.bn_11(un5)
        x11 = self.relu(x11)

        conv_layer = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=5, padding=2).to(device)
        x11 = x11.to(device)  # 将输入也移到 GPU

        output = conv_layer(x11)  # 前向传播
        return output

    @classmethod
    def from_tensorflow(cls, tf_path):
        """
        从TensorFlow预训练模型创建Embedder实例
        """
        model = cls()
        return convert_tf_weights(model, tf_path)


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层
        self.dconv_1 = nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2)
        self.bn_1 = nn.BatchNorm2d(32)

        # 第二层
        self.dconv_2 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)
        self.bn_2 = nn.BatchNorm2d(32)

        # 第三层
        self.dconv_3 = nn.Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2)
        self.bn_3 = nn.BatchNorm2d(64)

        # 第四层
        self.dconv_4 = nn.Conv2d(64, 64, kernel_size=5, stride=(1, 2), padding=2)
        self.bn_4 = nn.BatchNorm2d(64)

        # 第五层
        self.dconv_5 = nn.Conv2d(64, 128, kernel_size=5, stride=(1, 2), padding=2)
        self.bn_5 = nn.BatchNorm2d(128)

        # 第六层
        self.dconv_6 = nn.Conv2d(128, 128, kernel_size=5, stride=(1, 2), padding=2)
        self.bn_6 = nn.BatchNorm2d(128)

        # 激活函数
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Flatten层
        self.flatten = nn.Flatten()

        # 计算全连接层的输入维度
        # 假设输入是 (batch_size, 2, 512, 64)
        # 经过上述卷积层后的尺寸变化：
        # (2, 512, 64) -> (32, 256, 32) -> (32, 128, 16) -> (64, 128, 8)
        # -> (64, 128, 4) -> (128, 128, 2) -> (128, 128, 1)
        flatten_size = 128 * 128 * 1

        # 全连接层
        self.dense = nn.Linear(flatten_size, NUM_BITS)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 第一层
        x = self.dconv_1(x)
        x = self.bn_1(x)
        x = self.leaky_relu(x)

        # 第二层
        x = self.dconv_2(x)
        x = self.bn_2(x)
        x = self.leaky_relu(x)

        # 第三层
        x = self.dconv_3(x)
        x = self.bn_3(x)
        x = self.leaky_relu(x)

        # 第四层
        x = self.dconv_4(x)
        x = self.bn_4(x)
        x = self.leaky_relu(x)

        # 第五层
        x = self.dconv_5(x)
        x = self.bn_5(x)
        x = self.leaky_relu(x)

        # 第六层
        x = self.dconv_6(x)
        x = self.bn_6(x)
        x = self.leaky_relu(x)

        # Flatten
        x = self.flatten(x)

        # Dense + Sigmoid
        x = self.dense(x)
        x = self.sigmoid(x)

        return x

    def from_tensorflow(self, model_path):
        """从TensorFlow模型加载权重（如果需要的话）"""
        # 这里添加加载TensorFlow权重的代码
        return self

class AudioWatermarkModel(nn.Module):
    """
    完整的音频水印模型，包含嵌入器、攻击层和检测器
    """
    def __init__(self):
        super(AudioWatermarkModel, self).__init__()
        self.embedder = Embedder()
        self.attacks = nn.ModuleList([
            LowpassFilter(),
            AdditiveNoise(),
            CuttingSamples(),
            ButterworthFilter()
        ])
        self.detector = Detector()
        
    def forward(self, signal, message):
        # 嵌入水印
        embedded = self.embedder(signal, message)
        
        # 应用攻击
        attacked = embedded
        for attack in self.attacks:
            attacked = attack(attacked)
            
        # 检测水印
        detected = self.detector(attacked)
        
        return embedded, attacked, detected

def convert_tf_weights(model, tf_path):
    """
    将TensorFlow预训练权重转换并加载到PyTorch模型中
    
    Args:
        model: PyTorch模型实例 (Embedder或Detector)
        tf_path: TensorFlow模型路径
    """
    logger.info(f"正在从{tf_path}加载TensorFlow权重...")
    
    try:
        tf_model = tf.keras.models.load_model(tf_path)
        
        # 获取所有有权重的层
        tf_layers = [layer for layer in tf_model.layers if len(layer.get_weights()) > 0]
        torch_layers = [m for m in model.modules() if
                       isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d))]
        
        for tf_layer, torch_layer in zip(tf_layers, torch_layers):
            # 转换卷积层或转置卷积层
            if isinstance(torch_layer, (nn.Conv2d, nn.ConvTranspose2d)):
                weights = tf_layer.get_weights()[0]
                weights = np.transpose(weights, (3, 2, 0, 1))
                torch_layer.weight.data = torch.from_numpy(weights).float()
                
                if len(tf_layer.get_weights()) > 1:
                    bias = tf_layer.get_weights()[1]
                    torch_layer.bias.data = torch.from_numpy(bias).float()
            
            # 转换批归一化层
            elif isinstance(torch_layer, nn.BatchNorm2d):
                # print(tf_layer.shape)
                gamma = tf_layer.get_weights()[0]
                beta = tf_layer.get_weights()[1]
                mean = tf_layer.get_weights()[2]
                var = tf_layer.get_weights()[3]
                
                torch_layer.weight.data = torch.from_numpy(gamma).float()
                torch_layer.bias.data = torch.from_numpy(beta).float()
                torch_layer.running_mean.data = torch.from_numpy(mean).float()
                torch_layer.running_var.data = torch.from_numpy(var).float()
        
        logger.info("TensorFlow权重转换成功！")
        return model
        
    except Exception as e:
        logger.error(f"转换权重时出错: {e}")
        raise


def prepare_message():
    """
    准备水印消息以适配模型输入

    Args:
        message: 形状为 (batch_size, 512) 或 (512,) 的二进制消息
    Returns:
        torch.Tensor: 形状为 (batch_size, 16, 2, 512) 的张量
    """
    message_index = 3
    message = torch.from_numpy(MESSAGE_POOL[message_index])
    # if isinstance(message, np.ndarray):
    #     message = torch.from_numpy(message).float()
    # elif isinstance(message, torch.Tensor):
    #     message = message.float()
    # else:
    #     raise TypeError("消息必须是numpy数组或PyTorch张量")

    # 确保消息是2维的 [batch_size, 512]
    if message.dim() == 1:
        message = message.unsqueeze(0)  # [1, 512]

    batch_size = message.size(0)

    # 重塑为所需维度 [batch_size, 16, 2, 512]
    message = message.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, 512]
    message = message.expand(batch_size, 16, 2, -1)  # [batch_size, 16, 2, 512]

    return message.to(DEVICE)