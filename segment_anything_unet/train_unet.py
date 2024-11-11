import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, ScaleIntensity, Resize
from monai.metrics import compute_surface_dice
import numpy as np
from PIL import Image
from tqdm import tqdm
from monai.metrics import DiceMetric
from monai.metrics import compute_surface_dice
from monai.losses import DiceLoss


# 定义 MedicalDataset 数据集类
class MedicalDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_paths = []
        self.transforms = transforms
        # 遍历每个子文件夹，收集图像和标签路径
        for subfolder in os.listdir(data_dir):
            subfolder_path = os.path.join(data_dir, subfolder)
            if os.path.isdir(subfolder_path):
                images = sorted([os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith('.jpg')])
                labels = sorted([os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith('.npy')])
                assert len(images) == len(labels), f"Mismatch in images and labels in {subfolder}"
                self.data_paths.extend(zip(images, labels))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image_path, label_path = self.data_paths[idx]
        
        # 加载图像
        image = Image.open(image_path).convert("L")
        image = np.array(image, dtype=np.float32)
        
        # 加载标签
        label = np.load(label_path).astype(np.float32)
        
        # 添加单通道维度
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        # 转换和预处理
        if self.transforms:
            image = self.transforms(image)
            label = self.transforms(label)
            
        return {"image": image, "label": label}

# 训练和测试函数
def train_unet(data_dir, num_epochs=10, batch_size=4, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化 U-Net 模型
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    # 损失函数和优化器
    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 定义 Dice Metric
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # 数据变换
    train_transforms = Compose([
        ScaleIntensity(),
        Resize(spatial_size=(256, 256))
    ])
    
    # 加载数据集并划分为训练集和测试集
    full_dataset = MedicalDataset(data_dir, transforms=train_transforms)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 开始训练
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for batch_data in train_progress:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_progress.set_postfix(loss=epoch_loss / len(train_loader))
        
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss/len(train_loader):.4f}")
        
        # 初始化 DiceMetric 和 损失函数
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        loss_function = DiceLoss(sigmoid=True)

        # 测试模型
        model.eval()
        test_loss = 0
        dice_scores = []
        nsd_scores = []

        with torch.no_grad():
            test_progress = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Testing")
            for batch_data in test_progress:
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                outputs = model(inputs)
                
                # 计算批次的损失
                loss = loss_function(outputs, labels)
                test_loss += loss.item()

                # 逐样本计算 Dice 和 NSD 分数
                for i in range(outputs.shape[0]):
                    # 单个样本预测与标签
                    y_pred = (outputs[i] > 0.5).float()
                    y_true = labels[i]
                    
                    # 计算 Dice 分数
                    dice_score = dice_metric(y_pred=y_pred.unsqueeze(0), y=y_true.unsqueeze(0)).item()
                    dice_scores.append(dice_score)
                    
                    # 计算 NSD 分数
                    nsd_score = compute_surface_dice(y_pred=y_pred.unsqueeze(0), y=y_true.unsqueeze(0), class_thresholds=[1.0]).item()
                    nsd_scores.append(nsd_score)
                
                # 更新进度条
                test_progress.set_postfix(loss=test_loss / len(test_loader))

        # 计算平均 Dice 和 NSD 分数
        mean_dice = sum(dice_scores) / len(dice_scores)
        mean_nsd = sum(nsd_scores) / len(nsd_scores)

        print(f"Epoch {epoch+1}/{num_epochs}, Testing Loss: {test_loss/len(test_loader):.4f}, Mean Dice: {mean_dice:.4f}, Mean NSD: {mean_nsd:.4f}")

    # 保存模型权重
    torch.save(model.state_dict(), "Task09_Spleen_256_unet_model.pth")
    print("Model training complete and saved to 'unet_model.pth'.")

if __name__ == "__main__":
    data_dir = "/mnt/data/huangyifei/segment-anything-2/videos/Task09_Spleen_256"
    train_unet(data_dir, num_epochs=10, batch_size=4, learning_rate=1e-4)
