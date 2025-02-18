import torch
import cv2
import os
import numpy as np
from torchvision import transforms
import torch.nn as nn

# 字符映射字典（需与训练时一致）
characters = [chr(ord('a') + i) for i in range(26)]+ [str(i) for i in range(10)] 
idx2char = {idx: char for idx, char in enumerate(characters)}

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 与训练时相同的归一化
        ])

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

def remove_black_lines(image_path):
    # # 读取图像
    # img = cv2.imread(image_path)
    img = image_path
    # # 创建黑色像素掩膜（RGB都小于10）
    mask = np.all(img < [50, 50, 50], axis=2).astype(np.uint8) * 255
    # # 使用图像修复去除干扰线
    result = cv2.inpaint(img, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

    return result

def preprocess(image_path):
    """图像预处理流程"""
    # 读取图像并去除干扰线
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像：{image_path}")
  
    cleaned_img = remove_black_lines(img)

    # 转换为灰度图
    # gray_img = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)
    
    # 调整尺寸（确保与训练尺寸一致）
    resized_img = cv2.resize(cleaned_img, (90, 32))  # 宽度x高度
    
    # 转换为模型输入格式
    tensor_img = transform(resized_img)
    return tensor_img.unsqueeze(0)  # 添加batch维度

def predict(model,image_path):
    """执行预测"""
    try:
        # 预处理
        input_tensor = preprocess(image_path)
        
        # 推理
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # 获取预测结果
        _, preds = torch.max(outputs, dim=2)
        pred_str = ''.join([idx2char[idx.item()] for idx in preds[0]])
        
        return pred_str
    except Exception as e:
        print(f"预测失败：{str(e)}")
        return None
    
def convert_to_onnx(model, output_path):
    dummy_input = torch.randn(1, 3, 32, 90)
    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=['input'], output_names=['output'], opset_version=9
        # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

      
# 初始化模型（需与训练架构完全一致）
model = TinyCNN()  # 替换为实际使用的模型类
model.load_state_dict(torch.load(r"0.9997(3)_light_captcha_model.pth"))
model.eval()
convert_to_onnx(model,"cnn.onnx")

# all_files = [f for f in os.listdir(r"project\clean") if f.endswith(".jpg")]
# for image in all_files:
#     img=f"project/clean/{image}"
#     result = predict(model,img)
#     print(result)
