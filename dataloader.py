import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import glob

# 全局缩放函数（解决Windows多进程序列化问题）
def scale_to_neg_one(t):
    return (t * 2) - 1

class ImageDataset(Dataset):
    def __init__(self, root_dir, img_size=64, transform=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        
        # 获取所有图像路径（支持JPG/PNG）
        self.image_paths = []
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            self.image_paths += glob.glob(os.path.join(root_dir, '**', ext), recursive=True)
        
        if not self.image_paths:
            raise FileNotFoundError(f"在 {root_dir} 中未找到图像文件")
        print(f"找到 {len(self.image_paths)} 张图像")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image  # 仅返回图像张量，不返回标签[2](@ref)
        except Exception as e:
            print(f"加载失败: {img_path}, 错误: {e}")
            return torch.zeros(3, self.img_size, self.img_size)  # 返回占位符

def load_transformed_dataset(
    root_dir: str = "./data",
    img_size: int = 128,
    batch_size: int = 32,
    validation_split: float = 0.0
) -> DataLoader:
    # 数据转换流程
    data_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 确保固定尺寸[4](@ref)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(scale_to_neg_one)
    ])
    
    # 创建数据集实例
    full_dataset = ImageDataset(root_dir, img_size, transform=data_transforms)
    
    # Windows平台禁用多进程
    num_workers = 0 if os.name == 'nt' else 4
    
    # 数据集划分逻辑优化
    if 0 < validation_split < 1:
        train_size = int((1 - validation_split) * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size]
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        return train_loader, test_loader  
    else:
        return DataLoader(  # 单数据加载器
            full_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )



def show_tensor_image(image, save_path="denoised.png"):
    """
    仅保存纯净图像（无弹窗/无坐标轴/无白边）
    参数:
        image: 输入张量 [C, H, W] 或 [B, C, H, W]
        save_path: 图像保存路径（默认"denoised.png"）
    """
    # 移除批次维度（若存在）
    if image.dim() == 4: 
        image = image.squeeze(0)
    
    # 转换为[H, W, C]格式
    image = image.permute(1, 2, 0).cpu().numpy()
    
    # 检查数据范围并自动归一化
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.min() < 0 or image.max() > 1:
            image = (image - image.min()) / (image.max() - image.min())
    
    # 创建内存中的画布（不显示窗口）
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # 添加图像到画布
    ax.imshow(image)
    
    # 保存纯净图像（不显示窗口）
    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches='tight',
        pad_inches=0
    )
    
    # 关键修复：立即关闭资源且不显示窗口
    plt.close(fig)

# 修复使用示例
if __name__ == "__main__":
    # 加载数据集（根据是否划分验证集处理返回值）
    validation_split = 0.2
    loader = load_transformed_dataset(
        root_dir="./data",
        img_size=128,
        batch_size=32,
        validation_split=validation_split
    )
    
    # 根据返回值类型处理[4](@ref)
    if validation_split > 0:
        train_loader, val_loader = loader  # 解包两个加载器
        print("训练集加载器:", type(train_loader))
        print("验证集加载器:", type(val_loader))
        batch = next(iter(train_loader))  # 仅获取图像批次
    else:
        batch = next(iter(loader))  # 直接获取批次
    
    print(f"批次形状: {batch.shape}")  # 应为 [32, 3, 64, 64]
    show_tensor_image(batch[0])