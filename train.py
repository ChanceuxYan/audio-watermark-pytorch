import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import *
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WatermarkLoss(nn.Module):
    """
    水印模型的损失函数
    """
    def __init__(self):
        super(WatermarkLoss, self).__init__()
        self.mae = nn.L1Loss()
        self.bce = nn.BCELoss()
        self.encoder_loss_weight = 1.0
        self.decoder_loss_weight = 3.0
        
    def update_weights(self, step):
        """更新损失权重"""
        if step > 0 and step % 1400 == 0:
            self.encoder_loss_weight += 0.2
            self.decoder_loss_weight -= 0.2
        if step >= 14000:
            self.encoder_loss_weight = 2.5
            self.decoder_loss_weight = 0.5
    
    def forward(self, embedded, detected, original_signal, original_message, step):
        """
        计算总损失
        Args:
            embedded: 嵌入水印后的信号
            detected: 检测到的水印
            original_signal: 原始信号
            original_message: 原始水印消息
            step: 当前训练步骤
        """
        self.update_weights(step)
        
        encoder_loss = self.mae(embedded, original_signal)
        decoder_loss = self.bce(detected, original_message.squeeze())
        
        total_loss = (self.encoder_loss_weight * encoder_loss + 
                     self.decoder_loss_weight * decoder_loss)
        
        return total_loss, encoder_loss, decoder_loss

def train_step(model, data, optimizer, criterion, step):
    """
    单步训练
    """
    model.train()
    optimizer.zero_grad()
    
    signal, message = data
    signal = signal.to(DEVICE)
    message = message.to(DEVICE)
    
    # 前向传播
    embedded, attacked, detected = model(signal, message)
    
    # 计算损失
    loss, e_loss, d_loss = criterion(embedded, detected, signal, message, step)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    return loss.item(), e_loss.item(), d_loss.item()

def train(model, train_loader, num_epochs=NUM_EPOCHS):
    """
    训练模型
    Args:
        model: 水印模型
        train_loader: 训练数据加载器
        num_epochs: 训练轮数
    """
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = WatermarkLoss()
    
    logger.info("开始训练...")
    
    step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_encoder_loss = 0
        epoch_decoder_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:
            loss, e_loss, d_loss = train_step(model, batch, optimizer, criterion, step)
            
            epoch_loss += loss
            epoch_encoder_loss += e_loss
            epoch_decoder_loss += d_loss
            
            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'e_loss': f'{e_loss:.4f}',
                'd_loss': f'{d_loss:.4f}'
            })
            
            step += 1
        
        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        avg_e_loss = epoch_encoder_loss / len(train_loader)
        avg_d_loss = epoch_decoder_loss / len(train_loader)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs} - '
                   f'Loss: {avg_loss:.4f}, '
                   f'Encoder Loss: {avg_e_loss:.4f}, '
                   f'Decoder Loss: {avg_d_loss:.4f}')
        
        # 保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'{MODEL_SAVE_DIR}/model_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}') 