import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os
import random
from torch.utils.data import Dataset, DataLoader
import keyboard
import pyautogui
import time
from mss import mss  # 用于屏幕捕获
import math
import obsws_python as obs
import socket
import tempfile

host = socket.gethostbyname(socket.gethostname())
port = 8005

ws = obs.ReqClient(host=host,port=port)

pyautogui.FAILSAFE = False

# 确保必要的目录存在
ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(ROOT, 'models')
TRAIN_IMG_PATH = os.path.join(ROOT, 'train_img')
TEST_IMG_PATH = os.path.join(ROOT, 'test_img')
RESULTS_PATH = os.path.join(ROOT, 'results')
TEMP_DIR = tempfile.gettempdir()

for path in [MODELS_PATH, RESULTS_PATH]:
    os.makedirs(path, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 全局配置参数
CONFIG = {
    'image_size': (160, 90),  # 16:9宽屏尺寸
    'num_classes': 4,
    'num_anchors': 4,
    'confidence_threshold': 0.5,
    'nms_threshold': 0.45,
    'batch_size': 128,  # 减小batch_size以降低CPU负担
    'num_epochs': 50,
    'learning_rate': 0.001,
    # 类别名称映射
    'class_names': ['circle', 'slider', 'spinner', 'back']
}

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 調試用

class OsuNet(nn.Module):
    """Osu游戏对象分类网络 - 优化：简化网络结构，减少计算量"""
    def __init__(self, num_classes=CONFIG['num_classes']):
        super().__init__()
        self.num_classes = num_classes
        
        # 简化的卷积骨干网络，减少通道数和层数
        self.backbone = nn.Sequential(
            # 输入 3x160x90 -> 输出 16x80x45
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 输入 16x80x45 -> 输出 32x40x22
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 输入 32x40x22 -> 输出 64x20x11
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 输入 64x20x11 -> 输出 128x10x5
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 简化的分类头，减少神经元数量和dropout层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化，输出 [batch_size, 128, 1, 1]
            nn.Flatten(),  # 展平为 [batch_size, 128]
            nn.Linear(128, 64),  # 减少神经元数量
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)  # 直接输出分类结果
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """优化的权重初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 使用Kaiming初始化代替默认的正态分布，更适合ReLU激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        predictions = self.classifier(features)
        return predictions


def create_anchor_boxes(feature_size=None, image_size=CONFIG['image_size']):
    """创建锚点框，基于特征图尺寸"""
    anchors = []
    # 不同尺寸和宽高比的锚点框，考虑16:9比例
    base_sizes = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.2, 0.1), (0.1, 0.2)]  # 添加不同宽高比的锚框
    
    for size in base_sizes:
        w, h = size[0] * image_size[0], size[1] * image_size[1]
        anchors.append([w, h])
    
    return torch.tensor(anchors).float()  # [5, 2]


def decode_predictions(predictions, anchor_boxes, image_size=CONFIG['image_size'], num_classes=CONFIG['num_classes']):
    """
    将网络输出解码为实际的边界框
    predictions: [batch, A*(4+1+C), H, W]
    返回: 边界框, 置信度, 类别概率
    """
    batch_size, _, grid_h, grid_w = predictions.shape
    num_anchors = anchor_boxes.shape[0]
    
    # 重塑预测值 [batch, A, 5+C, H, W]
    pred = predictions.view(batch_size, num_anchors, -1, grid_h, grid_w)
    
    # 提取各个分量
    bbox_deltas = pred[:, :, :4, :, :]  # [batch, A, 4, H, W]
    objectness = torch.sigmoid(pred[:, :, 4:5, :, :])  # 物体置信度
    class_probs = torch.softmax(pred[:, :, 5:5+num_classes, :, :], dim=2)  # 类别概率 - 确保正确的切片
    
    # 计算网格中心坐标
    grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w))
    grid_x = grid_x.to(predictions.device).float()
    grid_y = grid_y.to(predictions.device).float()
    
    # 计算真实边界框坐标 - 分别考虑宽度和高度
    stride_h = image_size[1] / grid_h
    stride_w = image_size[0] / grid_w
    
    # 解码边界框坐标
    box_centers_x = (torch.sigmoid(bbox_deltas[:, :, 0, :, :]) + grid_x) * stride_w
    box_centers_y = (torch.sigmoid(bbox_deltas[:, :, 1, :, :]) + grid_y) * stride_h
    box_widths = torch.exp(bbox_deltas[:, :, 2, :, :]) * anchor_boxes[:, 0].view(1, -1, 1, 1)
    box_heights = torch.exp(bbox_deltas[:, :, 3, :, :]) * anchor_boxes[:, 1].view(1, -1, 1, 1)
    
    # 转换为左上角和右下角坐标
    x1 = box_centers_x - box_widths / 2
    y1 = box_centers_y - box_heights / 2
    x2 = box_centers_x + box_widths / 2
    y2 = box_centers_y + box_heights / 2
    
    # 确保边界框在图像范围内
    x1 = torch.clamp(x1, 0, image_size[0] - 1)
    y1 = torch.clamp(y1, 0, image_size[1] - 1)
    x2 = torch.clamp(x2, 0, image_size[0] - 1)
    y2 = torch.clamp(y2, 0, image_size[1] - 1)
    
    boxes = torch.stack([x1, y1, x2, y2], dim=2)  # [batch, A, 4, H, W]
    
    return boxes, objectness, class_probs


def non_max_suppression(boxes, scores, threshold=CONFIG['nms_threshold']):
    """非极大值抑制，过滤重叠的边界框"""
    if len(boxes) == 0:
        return []
    
    # 转换为numpy数组进行处理
    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    
    # 计算边界框面积
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 2]
    y2 = boxes_np[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # 按分数排序
    order = scores_np.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # 计算与其他边界框的重叠区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        # 计算IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 保留IoU小于阈值的边界框
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    
    return keep


def detect_objects(model, image_path, confidence_threshold=CONFIG['confidence_threshold'], 
                  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """在单张图像上进行对象检测"""
    model.eval()
    model.to(device)
    
    # 加载和预处理图像 - 修改为灰度图
    image = Image.open(image_path).convert('L')
    # 转换回RGB以保持通道数一致
    image = image.convert('RGB')
    original_size = image.size
    resized_image = transforms.Resize(CONFIG['image_size'])(image)
    image_tensor = transforms.ToTensor()(resized_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # 解码预测 - 确保传入正确的类别数
    anchor_boxes = create_anchor_boxes().to(device)
    boxes, objectness, class_probs = decode_predictions(predictions, anchor_boxes, CONFIG['image_size'][0])
    
    # 转换为平面格式
    batch_size, num_anchors, _, grid_h, grid_w = boxes.shape
    boxes = boxes.view(batch_size, -1, 4)  # [batch, A*H*W, 4]
    objectness = objectness.view(batch_size, -1)  # [batch, A*H*W]
    class_probs = class_probs.view(batch_size, -1, CONFIG['num_classes'])  # [batch, A*H*W, C]
    
    detected_boxes = []
    detected_scores = []
    detected_classes = []
    
    # 处理每个批次（这里只有一个图像）
    for b in range(batch_size):
        # 获取高置信度的检测
        high_conf_indices = (objectness[b] > confidence_threshold).nonzero().squeeze(1)
        
        if len(high_conf_indices) == 0:
            continue
        
        # 提取高置信度的边界框和分数
        high_conf_boxes = boxes[b][high_conf_indices]
        high_conf_scores = objectness[b][high_conf_indices]
        high_conf_classes = class_probs[b][high_conf_indices].argmax(dim=1)
        
        # 应用非极大值抑制
        keep_indices = non_max_suppression(high_conf_boxes, high_conf_scores)
        
        # 保存结果
        for idx in keep_indices:
            box = high_conf_boxes[idx]
            # 转换回原始图像尺寸
            scale_x = original_size[0] / CONFIG['image_size'][0]
            scale_y = original_size[1] / CONFIG['image_size'][1]
            box_scaled = [
                box[0].item() * scale_x,
                box[1].item() * scale_y,
                box[2].item() * scale_x,
                box[3].item() * scale_y
            ]
            detected_boxes.append(box_scaled)
            detected_scores.append(high_conf_scores[idx].item())
            detected_classes.append(high_conf_classes[idx].item())
    
    return detected_boxes, detected_scores, detected_classes


class OsuDataset(Dataset):
    """Osu游戏对象检测数据集"""
    def __init__(self, img_paths=None, labels=None, img_dir=None, transform=None, image_size=CONFIG['image_size'], augment=False):
        self.image_size = image_size
        self.num_classes = CONFIG['num_classes']
        self.class_names = CONFIG['class_names']
        self.img_paths = []
        self.labels = []
        
        # 定义增强的数据增强和预处理转换
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomRotation(15),  # 增加旋转角度
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),  # 随机裁剪和缩放
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 增强颜色抖动
                transforms.RandomHorizontalFlip(p=0.5),  # 增加水平翻转概率
                transforms.RandomVerticalFlip(p=0.2),  # 添加垂直翻转
                transforms.RandomGrayscale(p=0.1),  # 添加灰度变换
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # 添加高斯模糊
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            if self.transform is None:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        
        # 支持直接传入图像路径和标签
        if img_paths is not None:
            self.img_paths = img_paths
            self.labels = labels if labels is not None else [0] * len(self.img_paths)
        # 支持分类文件夹结构
        elif img_dir and os.path.exists(img_dir):
            self._load_images(img_dir)
        # 兼容旧版本的参数格式
        elif img_dir and isinstance(img_dir, list):
            self.img_paths = img_dir
            self.labels = labels if labels is not None else [0] * len(self.img_paths)
    
    def _load_images(self, img_dir):
        """加载图像路径和标签"""
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    self.img_paths.append(img_path)
                    
                    # 从文件夹名称获取标签
                    label = os.path.basename(root)
                    # 将标签映射为类别索引
                    try:
                        # 尝试将文件夹名作为类别索引
                        class_idx = int(label) - 1  # 假设文件夹是1,2,3...
                        # 确保类别索引在有效范围内
                        class_idx = max(0, min(class_idx, self.num_classes - 1))
                        self.labels.append(class_idx)
                    except ValueError:
                        # 如果无法转换为整数，尝试匹配类别名称
                        if label in self.class_names:
                            self.labels.append(self.class_names.index(label))
                        else:
                            # 默认使用0类别
                            self.labels.append(0)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            # 将图像转换为灰度图
            image = Image.open(img_path).convert('L')
            # 转换回RGB以保持通道数一致
            image = image.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # 返回图像和对应的标签索引
            return image, self.labels[idx]
        except Exception as e:
            print(f"加载图像 {img_path} 失败: {e}")
            # 返回默认图像和标签
            default_image = torch.zeros(3, self.image_size[1], self.image_size[0])
            return default_image, 0


def load_train_imgs(img_path):
    """加载训练图像和标签"""
    if not os.path.exists(img_path):
        print(f"警告: 训练图像路径不存在: {img_path}")
        return [], []
    
    # 定义图像转换 - 增强版本，增加多种数据增强
    transform = transforms.Compose([
        transforms.Resize(CONFIG['image_size']),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = OsuDataset(img_dir=img_path, transform=transform)
    imgs = [dataset[i][0] for i in range(len(dataset))]
    tags = dataset.labels
    
    return imgs, tags


def train_model(model, train_loader, num_epochs=CONFIG['num_epochs'], 
                             learning_rate=CONFIG['learning_rate'], device=None):
    """GPU优化的训练函数 - 优化spinner识别"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.train()
    
    print(f"🚀 开始GPU训练 on {device}")
    print(f"📊 模型设备: {next(model.parameters()).device}")
    
    # 使用优化的优化器配置，添加weight_decay正则化
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # 使用更平滑的余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 用于监控GPU使用情况
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # 调整类别权重，重点关注表现最差的类别
    # 基于评估结果，slider和spinner表现为0，需要大幅增加权重
# 设置类别权重，解决类别不平衡问题
    class_weights = torch.tensor([1.0, 3.0, 3.0, 3.0], device=device)  # 大幅增加slider、spinner和back类别的权重
    
    # 增加训练轮数，让模型有更多时间学习复杂特征
    num_epochs = max(num_epochs, 50)  # 确保至少训练50轮
    
    try:
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                # 确保数据在GPU上
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # 前向传播
                outputs = model(images)
                
                # 计算分类损失，使用类别权重
                loss = torch.nn.functional.cross_entropy(
                    outputs,  # 新模型直接输出分类结果 [batch_size, num_classes]
                    labels,
                    weight=class_weights  # 添加类别权重
                )
                
                # 反向传播
                optimizer.zero_grad(set_to_none=True)  # 更快的梯度清零
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                # 打印GPU内存使用情况（可选）
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**2
                    print(f"  批次 {batch_idx}, GPU内存: {memory_allocated:.1f}MB")
            
            # 更新学习率
            scheduler.step()
            
            avg_loss = epoch_loss / batch_count
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
            
            # 保存检查点
            if (epoch + 1) % 20 == 0:  # 更频繁地保存检查点
                model_path = os.path.join(MODELS_PATH, f'osu_model_temp.pth')
                torch.save(model.state_dict(), model_path)
                print(f"💾 模型已保存至: {model_path}")
    except KeyboardInterrupt:
        print("用户已中断训练")
    
    model_path = os.path.join(MODELS_PATH, f'osu_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"💾 模型已保存至: {model_path}")
    
    return model

def evaluate_model(model, test_dir, class_names=None):
    """评估模型在测试集上的性能"""
    model.eval()
    
    if not os.path.exists(test_dir):
        print(f"警告: 测试目录不存在: {test_dir}")
        return
    
    # 创建结果保存目录
    os.makedirs(os.path.join(RESULTS_PATH, 'detections'), exist_ok=True)
    
    # 获取测试图像
    test_images = []
    for file in os.listdir(test_dir):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            test_images.append(os.path.join(test_dir, file))
    
    # 对每张图像进行检测
    for img_path in test_images:
        boxes, scores, classes = detect_objects(model, img_path)
        
        # 可视化并保存结果
        img_name = os.path.basename(img_path)
        save_path = os.path.join(RESULTS_PATH, 'detections', f'detected_{img_name}')
        visualize_detection(img_path, boxes, scores, classes, class_names, save_path)
    
    print(f"评估完成！检测结果已保存至: {os.path.join(RESULTS_PATH, 'detections')}")


def generate_dataset(output_dir, num_samples_per_class=300, image_size=CONFIG['image_size']):
    """生成Osu游戏对象检测数据集"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义不同类别的输出目录
    classes_config = {
        '1': {'name': 'circle', 'generate_func': generate_circle},
        '2': {'name': 'slider', 'generate_func': generate_slider},
        '3': {'name': 'spinner', 'generate_func': generate_spinner}
    }
    
    total_samples = 0
    
    # 为每个类别生成样本
    for class_id, class_config in classes_config.items():
        class_dir = os.path.join(output_dir, class_id)
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"生成类别 {class_config['name']} 的样本...")
        for i in range(num_samples_per_class):
            # 创建空白图像
            image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
            
            # 调用对应的生成函数
            image = class_config['generate_func'](image, image_size)
            
            # 添加一些噪声
            if random.random() > 0.5:
                noise = np.random.randint(0, 50, (image_size[1], image_size[0], 3), dtype=np.uint8)
                image = cv2.add(image, noise)
            
            # 保存图像
            img_path = os.path.join(class_dir, f"{class_config['name']}_{i:05d}.png")
            cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            if (i + 1) % 100 == 0:
                print(f"  已生成 {i + 1}/{num_samples_per_class} 个{class_config['name']}样本")
        
        total_samples += num_samples_per_class
    
    print(f"数据集生成完成！共生成 {total_samples} 个样本，保存在: {output_dir}")

def generate_circle(image, image_size):
    """生成圆形目标"""
    # 随机生成圆形参数
    radius = random.randint(max(8, image_size[1]//10), min(30, image_size[0]//3))
    center_x = random.randint(radius, image_size[0] - radius)
    center_y = random.randint(radius, image_size[1] - radius)
    
    # 随机颜色（红色系列）
    color = (random.randint(200, 255), random.randint(0, 100), random.randint(0, 100))
    
    # 绘制圆形
    cv2.circle(image, (center_x, center_y), radius, color, -1)
    
    # 添加内部小圆作为细节
    if random.random() > 0.3:
        inner_radius = radius // 3
        inner_color = (255, 255, 255)
        cv2.circle(image, (center_x, center_y), inner_radius, inner_color, -1)
    
    return image

def generate_slider(image, image_size):
    """生成滑块目标"""
    # 随机生成滑块参数
    width = random.randint(10, 20)
    length = random.randint(30, 60)
    angle = random.randint(0, 180) * np.pi / 180  # 转换为弧度
    
    # 计算起点和终点
    center_x = random.randint(length//2 + 10, image_size[0] - length//2 - 10)
    center_y = random.randint(width//2 + 10, image_size[1] - width//2 - 10)
    
    # 滑块颜色（蓝色系列）
    color = (random.randint(0, 100), random.randint(0, 100), random.randint(200, 255))
    
    # 创建旋转矩阵
    rotation_mat = cv2.getRotationMatrix2D((center_x, center_y), angle * 180 / np.pi, 1)
    
    # 创建矩形
    x1 = center_x - length//2
    y1 = center_y - width//2
    x2 = x1 + length
    y2 = y1 + width
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    
    # 应用旋转
    rotated_pts = cv2.transform(pts.reshape(-1, 1, 2), rotation_mat).reshape(-1, 2)
    
    # 绘制旋转后的矩形
    cv2.fillPoly(image, [rotated_pts.astype(np.int32)], color)
    
    # 添加滑块轨道指示
    if random.random() > 0.5:
        # 计算轨道方向
        dx = np.cos(angle) * length * 0.8
        dy = np.sin(angle) * length * 0.8
        
        # 绘制轨道线
        cv2.line(image, (center_x, center_y), 
                (int(center_x + dx), int(center_y + dy)), 
                (255, 255, 255), 2)
    
    return image

def generate_spinner(image, image_size):
    """生成旋转器目标"""
    # 随机生成旋转器参数
    radius = random.randint(max(15, image_size[1]//8), min(40, image_size[0]//4))
    center_x = random.randint(radius + 10, image_size[0] - radius - 10)
    center_y = random.randint(radius + 10, image_size[1] - radius - 10)
    
    # 旋转器颜色（紫色系列）
    color = (random.randint(200, 255), random.randint(0, 100), random.randint(200, 255))
    
    # 绘制外圆
    cv2.circle(image, (center_x, center_y), radius, color, 3)
    
    # 绘制内圆
    inner_radius = radius // 2
    cv2.circle(image, (center_x, center_y), inner_radius, color, 2)
    
    # 绘制旋转指示线
    for i in range(4):
        angle = i * np.pi / 2 + random.uniform(0, np.pi/4)  # 随机旋转一点
        x1 = center_x + int(np.cos(angle) * inner_radius)
        y1 = center_y + int(np.sin(angle) * inner_radius)
        x2 = center_x + int(np.cos(angle) * radius)
        y2 = center_y + int(np.sin(angle) * radius)
        cv2.line(image, (x1, y1), (x2, y2), color, 2)
    
    # 添加中心点
    cv2.circle(image, (center_x, center_y), 3, (255, 255, 255), -1)
    
    return image


def main():
    """主函数 - 运行完整的多类别对象检测流程"""
    # 1. 创建模型 - 配置为3个类别
    model = OsuNet(num_classes=CONFIG['num_classes'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 2. 准备数据集
    TRAIN_IMG_PATH = 'train_img'
    TEST_IMG_PATH = 'test_img'
    
    # 显示当前配置信息
    print("===== 配置信息 =====")
    print(f"图像大小: {CONFIG['image_size']}")
    print(f"类别数量: {CONFIG['num_classes']}")
    print(f"类别名称: {CONFIG['class_names']}")
    print(f"批量大小: {CONFIG['batch_size']}")
    print(f"训练轮数: {CONFIG['num_epochs']}")
    print("==================\n")
    
    # 检查训练目录是否存在并包含类别文件夹
    need_generate = False
    if not os.path.exists(TRAIN_IMG_PATH):
        need_generate = True
    else:
        # 检查是否包含1、2、3三个类别文件夹
        has_all_classes = all(os.path.exists(os.path.join(TRAIN_IMG_PATH, str(i))) for i in [1, 2, 3])
        if not has_all_classes:
            print("训练目录缺少部分类别文件夹")
            need_generate = True
    
    # 如果需要，生成数据集
    if need_generate:
        print("生成训练数据集...")
        generate_dataset(TRAIN_IMG_PATH, num_samples_per_class=300)
    else:
        print(f"使用现有训练数据: {TRAIN_IMG_PATH}")
    
    # 3. 创建数据集和数据加载器 - 使用增强版本的OsuDataset
    print("加载训练数据...")
    
    # 使用增强的数据增强
    train_dataset = OsuDataset(img_dir=TRAIN_IMG_PATH, augment=True)
    
    if len(train_dataset) == 0:
        print("警告: 没有找到训练图像！")
        return
    
    # 显示数据分布统计
    from collections import Counter
    tag_counts = Counter(train_dataset.labels)
    print("数据分布:")
    for class_idx, count in tag_counts.items():
        class_name = CONFIG['class_names'][min(class_idx, len(CONFIG['class_names']) - 1)]
        print(f"  {class_name}: {count} 张图像")
    
    # 创建数据加载器，添加shuffle和num_workers
    num_workers=max(0,os.cpu_count()-2)  
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,  # 确保数据随机打乱
        num_workers=num_workers,  # 多线程加载数据，设置为CPU核心数的1-2倍
        pin_memory=True  # 加速GPU数据传输
    )
    
    # 5. 尝试加载现有模型（如果存在）
    model_path = os.path.join(MODELS_PATH, f'osu_model.pth')
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"✅ 已加载现有模型: {model_path}")
        except Exception as e:
            print(f"⚠️  加载模型失败: {e}，将从头开始训练")
    
    # 6. 训练模型 - 取消注释，确保模型被训练
    print("开始训练模型...")
    model = train_model(model, train_loader, device=device)
    
    # 7. 保存最终模型
    final_model_path = os.path.join(MODELS_PATH, f'osu_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"🎉 最终模型已保存至: {final_model_path}")
    
    # 8. 评估模型性能
    print("\n===== 开始模型评估 =====")
    # 直接在main.py中实现评估逻辑，避免subprocess环境问题
    from PIL import Image
    import torchvision.transforms as transforms
    from collections import Counter, defaultdict
    
    # 设置模型为评估模式
    model.eval()
    
    # 定义评估数据目录
    eval_dir = TRAIN_IMG_PATH
    
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize(CONFIG['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载评估图像
    eval_dataset = OsuDataset(img_dir=eval_dir, transform=transform)
    eval_loader = DataLoader(eval_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 评估变量
    correct = 0
    total = 0
    class_correct = [0] * CONFIG['num_classes']
    class_total = [0] * CONFIG['num_classes']
    confusion_matrix = [[0 for _ in range(CONFIG['num_classes'])] for _ in range(CONFIG['num_classes'])]
    
    # 无梯度评估
    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 类别级统计
            for i in range(len(labels)):
                label = labels[i]
                pred = predicted[i]
                confusion_matrix[label.item()][pred.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1
                class_total[label.item()] += 1
    
    # 计算指标
    overall_accuracy = 100 * correct / total
    print(f"总体准确率: {overall_accuracy:.2f}%")
    
    print("\n类别级准确率:")
    for i in range(CONFIG['num_classes']):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f"  {CONFIG['class_names'][i]}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    print("\n混淆矩阵:")
    print("          " + "  ".join([f"{name:>8}" for name in CONFIG['class_names']]))
    for i in range(CONFIG['num_classes']):
        row = [f"{confusion_matrix[i][j]:8d}" for j in range(CONFIG['num_classes'])]
        print(f"{CONFIG['class_names'][i]:>8}: {''.join(row)}")
    
    print("\n===== 训练和评估完成 =====")
    
    run(model)

def capture_screen():
    """捕获当前屏幕图像 - 优化：只捕获中心区域"""
    try:
        with mss() as sct:
            # 获取主显示器
            monitor = sct.monitors[1]  # 主显示器
            
            # 只捕获屏幕中心区域，减少处理数据量
            width = monitor['width']
            height = monitor['height']
            crop_width = int(width)
            crop_height = int(height)
            
            # 计算中心区域坐标
            left = int((width - crop_width) / 2)
            top = int((height - crop_height) / 2)
            
            # 创建自定义监控区域
            crop_monitor = {
                "left": left,
                "top": top,
                "width": crop_width,
                "height": crop_height
            }
            
            screenshot = sct.grab(crop_monitor)
            # 转换为numpy数组
            img = np.array(screenshot)
            # 转换为RGB格式（mss默认返回BGRA）
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            return img
    except Exception as e:
        print(f"屏幕捕获失败: {e}")
        # 返回一个空白图像作为备用
        return np.ones((720, 1280, 3), dtype=np.uint8) * 255

def preprocess_image(image):
    """预处理图像以供模型输入 - 优化：简化流程，直接使用OpenCV处理"""
    # 调整图像大小
    resized = cv2.resize(image, CONFIG['image_size'])
    # 直接转换为张量，避免PIL转换
    img_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    # 标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor

def parse_predictions(predictions, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """解析分类模型的预测结果 - 优化：简化计算，减少不必要的操作"""
    results = []
    
    # 分类模型输出 [batch_size, num_classes]
    # 获取第一个样本的预测结果
    class_scores = predictions[0]  # [num_classes]
    
    # 计算类别概率（使用softmax）
    class_probs = torch.softmax(class_scores, dim=0)  # [num_classes]
    
    # 找到最大概率的类别
    max_prob, predicted_class = torch.max(class_probs, dim=0)
    
    # 为不同类别设置不同的最小置信度阈值
    if predicted_class.item() == 0:  # circle
        min_confidence = 0.6  # circle需要较高置信度
    elif predicted_class.item() == 1:  # slider
        min_confidence = 0.7  # slider需要更高置信度，减少误判
    elif predicted_class.item() == 2:  # spinner
        min_confidence = 0.8  # spinner需要最高置信度
    elif predicted_class.item() == 3:
        min_confidence = 0.7
    else:
        min_confidence = CONFIG['confidence_threshold']
    
    if max_prob.item() < min_confidence:
        return results  # 置信度不足，返回空列表
    
    # 简化边界框计算，直接返回中心区域
    image_w, image_h = CONFIG['image_size']
    
    # 使用固定的边界框大小，减少计算
    center_w = image_w * 0.5
    center_h = image_h * 0.5
    x1 = int((image_w - center_w) / 2)
    y1 = int((image_h - center_h) / 2)
    x2 = int(x1 + center_w)
    y2 = int(y1 + center_h)
    
    # 创建边界框
    box = [x1, y1, x2, y2]
    
    # 添加到结果列表
    results.append({
        'box': box,
        'score': max_prob.item(),
        'class': predicted_class.item(),
        'class_name': CONFIG['class_names'][predicted_class.item()]
    })
    
    return results

def visualize_results(screen, results):
    """在屏幕图像上可视化检测结果"""
    # 创建屏幕的副本以避免修改原始图像
    display_img = screen.copy()
    screen_height, screen_width = display_img.shape[:2]
    
    # 类别颜色映射
    class_colors = {
        0: (0, 0, 255),      # circle - 红色
        1: (0, 255, 0),      # slider - 绿色
        2: (255, 0, 0)       # spinner - 蓝色
    }
    
    # 绘制检测结果
    for result in results:
        # 缩放边界框以匹配原始屏幕尺寸
        box = result['box']
        x1 = int(box[0] * screen_width / CONFIG['image_size'][0])
        y1 = int(box[1] * screen_height / CONFIG['image_size'][1])
        x2 = int(box[2] * screen_width / CONFIG['image_size'][0])
        y2 = int(box[3] * screen_height / CONFIG['image_size'][1])
        
        # 获取类别和分数
        class_id = result['class']
        score = result['score']
        class_name = CONFIG['class_names'][class_id] if class_id < len(CONFIG['class_names']) else f'Class {class_id}'
        color = class_colors.get(class_id, (255, 255, 0))  # 默认黄色
        
        # 绘制边界框
        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
        
        # 添加标签
        label = f'{class_name}: {score:.2f}'
        cv2.putText(display_img, label, (x1, max(y1-10, 10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 显示结果
    cv2.imshow('Osu Object Detection', display_img)
    cv2.waitKey(1)  # 保持窗口响应

def detect_object_position(screen):
    """使用图像处理技术检测屏幕中物体的实际位置"""
    # 转换为灰度图
    gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    
    # 应用阈值处理，突出明亮的物体
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找到最大的轮廓（假设是我们要检测的物体）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 计算中心点
        center_x = x + w // 2
        center_y = y + h // 2
        
        return center_x, center_y
    
    # 如果没有找到轮廓，返回屏幕中心
    return pyautogui.position()

def run(model: nn.Module):
    """运行实时检测循环"""
    is_run = False
    should_exit = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    model.to(device)
    model.eval()
    
    print("实时检测已启动！按 'q' 键切换状态，长按 'q' 2秒退出")
    
    screen_width, screen_height = pyautogui.size()
    slider_hold = False
    spinner_hold = False
    
    # 定义q键按下处理函数
    def on_q_press(event):
        nonlocal is_run, slider_hold, spinner_hold, should_exit
        # 只处理按下事件
        if event.name == 'q' and event.event_type == 'down':
            # 记录按键时间
            start_time = time.time()
            
            # 检查是否是长按
            while keyboard.is_pressed('q'):
                if time.time() - start_time > 2:
                    # 长按2秒退出
                    print("检测到长按，程序退出")
                    should_exit = True
                    return 'quit'
                time.sleep(0.01)
            
            # 如果不是长按，视为短按
            if not should_exit:  # 确保只在非退出情况下切换状态
                if is_run:
                    is_run = False
                    print("检测已暂停，再次按 'q' 继续")
                    cv2.destroyAllWindows()
                    # 确保释放鼠标
                    pyautogui.mouseUp()
                    slider_hold = False
                    spinner_hold = False
                else:
                    is_run = True
        return None
    
    # 注册按键事件处理器
    keyboard.hook(on_q_press)
    
    try:
        while not should_exit:
            if is_run:
                # 1. 捕获当前屏幕
                screen = capture_screen()
                
                #移除黑色背景
                screen[screen < 10] = 0
                # 确保是3通道图像
                screen = cv2.cvtColor(screen, cv2.COLOR_GRAY2RGB)
                cv2.imwrite(os.path.join(TEMP_DIR,"temp_osu_ai_can.png"),screen)
                ws.set_input_settings(
                    name="AICam",
                    settings={
                        "file":os.path.join(TEMP_DIR,"temp_osu_ai_can.png")
                    },
                    overlay=True
                )
                
                # 2. 预处理图像
                processed_img = preprocess_image(screen).to(device)
                
                # 3. 进行推理
                with torch.no_grad():
                    predictions = model(processed_img.unsqueeze(0))
                
                # 4. 解析预测结果
                results = parse_predictions(predictions, device)
                
                if results:
                    # 直接使用第一个结果，不需要排序
                    best_result = results[0]
                    current_class = best_result['class']
                    current_score = best_result['score']
                    
                    # 使用图像处理检测实际物体位置
                    object_x, object_y = detect_object_position(screen)
                    object_x = int(object_x * screen_width / screen.shape[1])
                    object_y = int(object_y * screen_height / screen.shape[0])
                    
                    # 重置鼠标状态，如果当前没有检测到高置信度的目标
                    if current_score < 0.7:
                        if slider_hold:
                            pyautogui.mouseUp()
                            slider_hold = False
                        if spinner_hold:
                            pyautogui.mouseUp()
                            spinner_hold = False
                    else:
                        # 处理不同类型的目标
                        if current_class == 0:  # circle
                            # 确保其他状态被重置
                            if slider_hold:
                                pyautogui.mouseUp()
                                slider_hold = False
                            
                            try:
                                pyautogui.click(object_x, object_y)
                            except Exception:
                                pass
                        
                        elif current_class == 1:  # slider
                            # 确保spinner状态被重置
                            if spinner_hold:
                                pyautogui.mouseUp()
                                spinner_hold = False
                            
                            try:
                                if not slider_hold:
                                    pyautogui.mouseDown(object_x, object_y)
                                    slider_hold = True
                                else:
                                    pyautogui.moveTo(object_x, object_y)
                            except Exception:
                                if slider_hold:
                                    pyautogui.mouseUp()
                                    slider_hold = False
                        
                        elif current_class == 2:  # spinner
                            # 确保slider状态被重置
                            if slider_hold:
                                pyautogui.mouseUp()
                                slider_hold = False
                            
                            try:
                                if not spinner_hold:
                                    pyautogui.mouseDown(object_x, object_y)
                                    spinner_hold = True
                                else:
                                    # 实现快速旋转 - 提高旋转速度
                                    current_time = 90  # 更高的旋转速度
                                    circle_radius = 50  # 增大旋转半径
                                    spin_x = int(object_x + circle_radius * math.sin(current_time))
                                    spin_y = int(object_y + circle_radius * math.cos(current_time))
                                    
                                    # 移动鼠标，使用移动持续时间为0以获得最快速度
                                    pyautogui.moveTo(spin_x, spin_y)
                                    current_time *= 1.5
                            except Exception:
                                if spinner_hold:
                                    pyautogui.mouseUp()
                                    spinner_hold = False
                        elif current_class == 3:
                            pass
                        else:
                            # 重置所有状态
                            pyautogui.mouseUp()
                            slider_hold = False   
                            spinner_hold = False 
                else:
                    # 没有检测到任何目标，重置鼠标状态
                    if slider_hold:
                        pyautogui.mouseUp()
                        slider_hold = False
                    if spinner_hold:
                        pyautogui.mouseUp()
                        spinner_hold = False
                time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        # 注销键盘钩子
        keyboard.unhook_all()
        # 确保释放鼠标
        pyautogui.mouseUp()
        cv2.destroyAllWindows()
        print("实时检测已结束")

# 如果作为主程序运行，执行主函数
if __name__ == '__main__':
    main()
