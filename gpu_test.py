import torch
import torch.nn as nn

CONFIG = {
    'image_size': (160, 90),  # 16:9宽屏尺寸
    'num_classes': 4,
    'num_anchors': 4,
    'confidence_threshold': 0.5,
    'nms_threshold': 0.45,
    'batch_size': 1024,
    'num_epochs': 500,
    'learning_rate': 0.001,
    # 类别名称映射
    'class_names': ['circle', 'slider', 'spinner', 'back']
}

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

def check_memory(batch_size=int(input("请输入batch_size: "))):
    # 模拟一个样本的大小
    model = OsuNet().cuda()
    
    input_tensor = torch.randn(batch_size, 3, 224, 224).cuda()
    
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测量实际显存使用
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    initial_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"初始显存占用: {initial_mem:.2f} GB")
    
    # 前向传播
    try:
        output = model(input_tensor)
        forward_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"前向后显存: {forward_mem:.2f} GB")
        
        # 反向传播
        loss = output.sum()
        loss.backward()
        
        backward_mem = torch.cuda.memory_allocated() / 1024**3
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"反向后显存: {backward_mem:.2f} GB")
        print(f"峰值显存: {peak_mem:.2f} GB")
        print(f"可用显存剩余: {4.0 - peak_mem:.2f} GB")
        
        if peak_mem > 3.2:  # 80% of 4GB
            print("⚠️  Warning: 接近显存上限!")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ OOM错误! batch_size={batch_size} 太大")
        else:
            raise e

check_memory()