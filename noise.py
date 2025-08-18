import torch
import torch.nn.functional as F


def get_index_from_list(vals, time_step, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = time_step.shape[0]
    out = vals.gather(-1, time_step.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(time_step.device)



# 扩散过程的总时间步数
T = 1000
# 生成噪声调度参数 beta，从 0.0001 线性增加到 0.01
beta = torch.linspace(0.0001, 0.01, T)
# 计算 alpha，表示每一步保留的信号比例
alpha = 1.0 - beta
# 计算 alpha 的累积乘积，表示从初始时间步到当前时间步的信号保留比例
alpha_cumprod = torch.cumprod(alpha, axis=0)
# 对 alpha_cumprod 进行填充，确保第一个时间步的 alpha_cumprod_prev 为 1.0
alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)
# 计算 alpha 的平方根的倒数，用于反向扩散过程
sqrt_recip_alphas = torch.sqrt(1.0 / alpha)
# 计算 alpha_cumprod 的平方根，用于正向扩散过程
sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
# 计算 1.0 - alpha_cumprod 的平方根，用于噪声生成
sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
# 计算后验方差，用于反向扩散过程中的噪声估计
posterior_variance = beta * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)

