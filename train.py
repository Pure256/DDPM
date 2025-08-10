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
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.mse_loss(noise, noise_pred)

def main():
    model = Unet()
    T = 300
    BATCH_SIZE = 128
    epochs = 100
    
    os.makedirs("./trained_models/checkpoints", exist_ok=True)
    
    dataloader = load_transformed_dataset(batch_size=BATCH_SIZE)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    
    for epoch in tqdm(range(epochs), desc="Total Progress", position=0):  
        epoch_loss = 0.0
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
            optimizer.step()
            
            epoch_loss += loss.item()
            # 实时更新batch进度信息
            batch_bar.set_postfix(
                batch_loss=f"{loss.item():.4f}",
                avg_loss=f"{epoch_loss/(batch_idx+1):.4f}"
            )
            
        # 更新epoch级信息
        tqdm.write(f"Epoch {epoch} | Avg Loss: {epoch_loss/len(dataloader):.4f}")
        
        # 每10个epoch保存一次模型（包含优化器状态）[3,5,7](@ref)
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }
            checkpoint_path = f"./trained_models/checkpoints/ddpm_checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"保存模型检查点: {checkpoint_path}")
    
    # 最终模型保存
    final_model_path = "./trained_models/ddpm_final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"训练完成，最终模型已保存至: {final_model_path}")
    
if __name__ == "__main__":
    main()