import math
import torch
from torch import nn


class Block(nn.Module):
    """
    基础卷积块，用于UNet的下采样和上采样路径
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        time_emb_dim (int): 时间嵌入的维度
        up (bool): 是否为上采样块（默认False）
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        super().__init__()
        # 时间嵌入的线性映射层
        self.time_emb = nn.Linear(time_emb_dim, out_channels)
        
        # 上采样和下采样的卷积配置
        
        if up:
            # 上采样时输入通道数为2倍（因为要拼接跳跃连接）
            self.conv1 = nn.Conv2d(2 * in_channels, out_channels, 3, padding=1)
            # 上采样使用转置卷积
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)
        else:
            # 下采样时输入通道数不变
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            # 下采样使用普通卷积
            self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
            
        # 第二层卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # 批归一化层
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.BN2 = nn.BatchNorm2d(out_channels)
        # ReLU激活函数
        self.relu = nn.ReLU()

    def forward(self, x, t):
        """
        前向传播
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_channels, height, width)
            t (torch.Tensor): 时间嵌入张量，形状为 (batch_size, time_emb_dim)
        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, out_channels, height // 2, width // 2)
        """
        # 第一层卷积 + 批归一化 + ReLU
        h = self.BN1(self.relu(self.conv1(x)))  # 形状: (batch_size, out_channels, height, width)
        
        # 时间嵌入处理
        time_emb = self.relu(self.time_emb(t))   # 形状: (batch_size, out_channels)
        # 拓展维度以匹配空间维度
        time_emb = time_emb[(...,) + (None,) * 2]  # 形状: (batch_size, out_channels, 1, 1)
        # 广播相加到特征图
        h = h + time_emb                         # 形状: (batch_size, out_channels, height, width)
        
        # 第二层卷积 + 批归一化 + ReLU
        h = self.BN2(self.relu(self.conv2(h)))    # 形状: (batch_size, out_channels, height, width)
        
        # 下采样或上采样变换
        return self.transform(h)                  # 形状: (batch_size, out_channels, height // 2, width // 2)


class PositionEmbedding(nn.Module):
    """
    生成时间步的正弦位置嵌入
    Args:
        dim (int): 嵌入的维度
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        前向传播，生成位置嵌入
        Args:
            time (torch.Tensor): 输入时间步，形状为 (batch_size,)
        Returns:
            torch.Tensor: 位置嵌入，形状为 (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        # 计算嵌入参数
        embeddings = math.log(10000) / (half_dim - 1)
        # 生成正弦和余弦嵌入
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)  # 形状: (half_dim,)
        embeddings = time[:, None] * embeddings[None, :]                              # 形状: (batch_size, half_dim)
        # 拼接正弦和余弦嵌入
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)          # 形状: (batch_size, dim)
        return embeddings


class Unet(nn.Module):
    """
    UNet模型，用于扩散模型的前向和反向过程
    Args:
        None
    """
    def __init__(self):
        super().__init__()
        # 输入图像的通道数（RGB为3）
        image_channels = 3
        # 下采样路径的通道数配置
        down_channels = (64, 128, 256, 512, 1024)
        # 上采样路径的通道数配置
        up_channels = (1024, 512, 256, 128, 64)
        # 输出图像的通道数（与输入一致）
        out_dim = 3
        # 时间嵌入的维度
        time_emb_dim = 32

        # 时间嵌入层
        self.time_emb = nn.Sequential(
            PositionEmbedding(time_emb_dim),  # 生成正弦位置嵌入
            nn.Linear(time_emb_dim, time_emb_dim),  # 线性映射
            nn.ReLU()  # ReLU激活
        )

        # 初始卷积层
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # 下采样块列表
        self.downs = nn.ModuleList(
            [Block(down_channels[i], down_channels[i + 1], time_emb_dim) for i in range(len(down_channels) - 1)]
        )

        # 上采样块列表
        self.ups = nn.ModuleList(
            [Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True) for i in range(len(up_channels) - 1)]
        )

        # 输出卷积层
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        """
        UNet的前向传播
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, channels, height, width)
            timestep (torch.Tensor): 时间步张量，形状为 (batch_size,)
        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, out_dim, height, width)
        """
        # 时间嵌入
        t = self.time_emb(timestep)  # 形状: (batch_size, time_emb_dim)

        # 初始卷积
        x = self.conv0(x)            # 形状: (batch_size, down_channels[0], height, width)

        # 下采样
        residual_inputs = []  # 保存跳跃连接的输入
        for down in self.downs:
            x = down(x, t)           # 形状: (batch_size, down_channels[i + 1], height // 2, width // 2)
            residual_inputs.append(x)  # 保存当前层的输出

        # 上采样
        for up in self.ups:
            # 取出下采样路径的跳跃连接
            residual_x = residual_inputs.pop()  # 形状: (batch_size, up_channels[i], height // 2, width // 2)
            # 拼接当前特征和跳跃连接
            x = torch.cat((x, residual_x), dim=1)  # 形状: (batch_size, 2 * up_channels[i], height // 2, width // 2)
            # 上采样块处理
            x = up(x, t)            # 形状: (batch_size, up_channels[i + 1], height, width)

        # 输出卷积
        return self.output(x)        # 形状: (batch_size, out_dim, height, width)


def main():
    """
    测试UNet模型
    - 初始化模型
    - 生成随机输入和时间步
    - 执行前向传播并打印输入输出形状
    """
    # 选择设备（GPU优先）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化UNet模型
    model = Unet().to(device)
    print(f"Model initialized on {device}.")

    # 生成随机输入和时间步
    batch_size = 32
    image_size = 128
    x = torch.randn(batch_size, 3, image_size, image_size).to(device)  # 形状: (batch_size, 3, image_size, image_size)
    timestep = torch.tensor([10], dtype=torch.long).to(device)         # 形状: (1,)

    # 前向传播（禁用梯度计算）
    with torch.no_grad():
        output = model(x, timestep)  # 形状: (batch_size, out_dim, image_size, image_size)
    
    # 打印输入输出形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()



