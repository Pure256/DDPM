import os
import torch
from noise import *
from model import Unet
from dataloader import load_transformed_dataset
import torch.nn.functional as F
from torch.optim import Adam
import logging
from tqdm import tqdm 

logging.basicConfig(level=logging.INFO)

def forward_diffusion_sample(x_0, time_step, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alpha_cumprod, time_step, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alpha_cumprod, time_step, x_0.shape)
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(
        device
    ), noise.to(device)

def get_loss(model, x_0, t, device):
    # 加噪样本和真实噪声
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)  
    noise_pred = model(x_noisy, t)  # 预测噪声
    # 计算基础MSE损失
    base_loss = F.mse_loss(noise_pred, noise, reduction="none")
    base_loss = base_loss.mean(dim=list(range(1, len(base_loss.shape))))  # 按样本维度平均
    # 计算SNR(t)
    alpha_cumprod_t = get_index_from_list(alpha_cumprod, t, x_0.shape)  # 获取当前t的ᾱ_t
    snr_t = alpha_cumprod_t / (1 - alpha_cumprod_t)  # SNR(t) = ᾱ_t / (1 - ᾱ_t)
    # Min-SNR-5加权：γ=5，抑制简单任务权重
    snr_weight = torch.clamp(snr_t, max=5.0)  # w_t = min(SNR(t), 5)
    weighted_loss = base_loss * snr_weight
    return weighted_loss.mean()  # 返回加权平均损失

def main():
    model = Unet()
    T = 1000
    BATCH_SIZE = 128
    epochs = 500
    
    os.makedirs("./trained_models/checkpoints", exist_ok=True)
    
    dataloader = load_transformed_dataset(batch_size=BATCH_SIZE)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model.to(device)
    
    # 设置检查点路径
    checkpoint_path = "ddpm_checkpoint_epoch_100.pth"
    
    if os.path.exists(checkpoint_path):
        logging.info(f"发现检查点文件: {checkpoint_path}，准备继续训练")
        
        checkpoint = torch.load(
            checkpoint_path, 
            weights_only=True,  # 显式启用安全加载
            map_location=device
        )
        
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict, strict=False)
        
        optimizer = Adam(model.parameters(), lr=1e-5)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    for epoch in tqdm(range(epochs), desc="Total Progress", position=0):
        epoch_loss = 0.0
        model.train()  
        
        batch_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1}/{epochs}", 
            position=1, 
            leave=False,
            total=len(dataloader)
        )
        
        for batch_idx, batch in enumerate(batch_bar):
            optimizer.zero_grad()
            batch = batch.to(device)
            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model, batch, t, device=device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_bar.set_postfix(
                batch_loss=f"{loss.item():.4f}",
                avg_loss=f"{epoch_loss/(batch_idx+1):.4f}"
            )
            
        tqdm.write(f"Epoch {epoch} | Avg Loss: {epoch_loss/len(dataloader):.4f}")
        
        if (epoch + 1) % 50 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }
            save_path = f"./trained_models/checkpoints/ddpm_checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint, save_path)
            logging.info(f"保存检查点: {save_path}")
    
    # 最终模型保存
    final_model_path = "./trained_models/ddpm_final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"训练完成，最终模型已保存至: {final_model_path}")
    
if __name__ == "__main__":
    main()