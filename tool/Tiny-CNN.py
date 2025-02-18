import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np

def remove_black_lines(image_path):
    # # 读取图像
    img = cv2.imread(image_path)
    # # 创建黑色像素掩膜（RGB都小于10）
    mask = np.all(img < [50, 50, 50], axis=2).astype(np.uint8) * 255
    # # 使用图像修复去除干扰线
    result = cv2.inpaint(img, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

    return result

# 1. 数据集类定义
class CaptchaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.filenames = [f for f in os.listdir(data_dir) if f.endswith(('.jpg','.png'))]
        
        # 创建字符到索引的映射
        self.char2idx = {chr(ord('a')+i): i for i in range(26)}
        for i in range(10):
            self.char2idx[str(i)] = i + 26

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.filenames[idx])
        # 去除干扰线并转为灰度图
        img = remove_black_lines(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 转换张量
        if self.transform:
            img = self.transform(img)
        
        # 从文件名获取标签
        label_str = self.filenames[idx].split('_')[0]
        label = [self.char2idx[c] for c in label_str]
        return img, torch.tensor(label)

# 2. 轻量CNN模型（<1MB）
class TinyCNN(nn.Module):
    def __init__(self, num_chars=4, num_classes=36):
        super(TinyCNN, self).__init__()
        self.features = nn.Sequential(
            # 输入: 1x32x90
            nn.Conv2d(3, 8, 3, padding=1),  # 16x32x90
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16x45
            
            nn.Conv2d(8, 16, 3, padding=1),  # 32x16x45
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x8x22
            
            nn.Conv2d(16, 32, 3, padding=1),  # 64x8x22
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 11))  # 64x4x11
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32*4*11, 128),
            nn.Dropout(0.3),
            nn.LayerNorm(128),  # 🌟 层归一化
            nn.Linear(128, num_chars*num_classes)
        )
        self.num_chars = num_chars
        self.num_classes = num_classes

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.view(-1, self.num_chars, self.num_classes)

# 4. 训练循环
def train(model, train_dataloader, val_dataloader, epochs=300, save_threshold=0.9985):
    model.train()  # 设置模型为训练模式
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)  # 定义优化器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, steps_per_epoch=len(train_dataloader), epochs=epochs
    )  # 定义学习率调度器

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_dataloader:
            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs)  # 前向传播

            # 计算四个字符的损失
            loss = sum(criterion(outputs[:, i, :], labels[:, i]) for i in range(4))
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            scheduler.step()  # 更新学习率

            # 计算准确率
            _, predicted = torch.max(outputs, 2)
            correct += (predicted == labels).all(1).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

        # 打印训练信息
        train_loss = total_loss / total
        train_acc = correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4%}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 验证模型
        val_loss, val_acc = validate(model, val_dataloader, criterion)
        print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4%}")

        # 保存模型
        if val_acc > save_threshold:
            torch.save(model.state_dict(), f"{val_acc:.6f}_light_captcha_model.pth")
            print(f"Model saved with validation accuracy: {val_acc:.4%}")


# 5. 验证函数
def validate(model, dataloader, criterion):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # 关闭梯度计算
        for inputs, labels in dataloader:
            outputs = model(inputs)  # 前向传播

            # 计算损失
            loss = sum(criterion(outputs[:, i, :], labels[:, i]) for i in range(4))
            total_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 2)
            correct += (predicted == labels).all(1).sum().item()
            total += labels.size(0)

    val_loss = total_loss / total
    val_acc = correct / total
    return val_loss, val_acc

if __name__ == "__main__":
# 计算参数量
model = TinyCNN()
print("模型参数量：", sum(p.numel() for p in model.parameters()))  # 约802.6KB

# 3. 训练配置
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = CaptchaDataset('path to your train data', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataset = CaptchaDataset('path to your validate data', transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=True)

# 启动训练
train(model,train_dataloader, val_dataloader, epochs=300, save_threshold=0.9985)

#把模型转为onnx模型
def convert_to_onnx(model, output_path):
    dummy_input = torch.randn(1, 3, 32, 90)
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 
                      'output': {0: 'batch_size'}}
    )

