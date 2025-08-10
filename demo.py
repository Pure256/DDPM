import torch
from noise import *
import matplotlib.pyplot as plt
from dataloader import show_tensor_image
from model import Unet
import os



@torch.no_grad()
def sample_timestep(model, x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(beta, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alpha_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(model, device, img_size, T):
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15, 15))  # 调整为合理尺寸[8](@ref)
    
    num_images = 5
    stepsize = T // num_images  # 确保整数步长
    
    for i in reversed(range(0, T)):
        t = torch.tensor([i], device=device, dtype=torch.long)
        img = sample_timestep(model, img, t)
        img = torch.clamp(img, -1.0, 1.0)
        
        if i % stepsize == 0:
            plt.subplot(1, num_images, (i // stepsize) + 1)
            show_tensor_image(img.detach().cpu())  # 无需squeeze
    
    plt.tight_layout()


if __name__ == "__main__":
    img_size = 128
    T = 300
    model = Unet()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    checkpoint_path = "ddpm_checkpoint_epoch_10.pth"
    
    # 验证文件是否存在[3](@ref)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    # 正确加载检查点[1,4,7](@ref)
    checkpoint = torch.load(
        checkpoint_path, 
        weights_only=True,  # 显式启用安全加载
        map_location=device
    )
    
    # 修复state_dict键名不匹配问题[4](@ref)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 非严格模式加载[5](@ref)
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)

    sample_plot_image(model=model, device=device, img_size=img_size, T=T)
