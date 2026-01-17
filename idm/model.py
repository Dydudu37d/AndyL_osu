import torch
import torch.nn as nn
import torchvision.models as models
from config import Config

class OsuIDM(nn.Module):
    def __init__(self):
        super(OsuIDM, self).__init__()
        
        print(f"🚀 Initializing OsuIDM (ResNet50-Dilated) for {Config.IMG_SIZE}x{Config.IMG_SIZE} Input...")
        
        # 1. 换回 ResNet50 (它是 Bottleneck 架构，支持 Dilation)
        # replace_stride_with_dilation=[False, True, True]
        # 效果：
        # Layer2: 保持 stride=2 (下采样) -> 此时分辨率 16x16
        # Layer3: 变为 stride=1, dilation=2 -> 分辨率保持 16x16
        # Layer4: 变为 stride=1, dilation=4 -> 分辨率保持 16x16
        # 最终输出特征图大小: 16x16 (完美符合我们的高精度需求)
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(
            weights=weights, 
            replace_stride_with_dilation=[False, True, True]
        )
        
        # 2. 魔改第一层 (适应 6 通道灰度输入)
        original_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=Config.INPUT_CHANNELS, # 6
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        
        with torch.no_grad():
            # RGB 均值初始化
            avg_weight = torch.mean(original_conv1.weight, dim=1, keepdim=True)
            new_conv1.weight.data = avg_weight.repeat(1, Config.INPUT_CHANNELS, 1, 1)
            
        # 3. 组装 Encoder
        # 既然用了 dilation 保持分辨率，我们可以把 MaxPool 加回来，保证网络的感受野足够大
        # 结构：Input(128) -> Conv1(64) -> MaxPool(32) -> Layer1(32) -> Layer2(16) -> Layer3(16) -> Layer4(16)
        self.encoder = nn.Sequential(
            new_conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool, # 加回 MaxPool，因为后面 dilation 足够维持 16x16
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        
        # 4. 颈部降维
        # ResNet50 Layer4 输出通道是 2048
        # Feature Map: 2048 x 16 x 16
        # 这种高分辨率+深通道，必须先大力降维
        self.compressor = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1), # 2048 -> 256
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True) 
        )
        
        # Flatten 后维度: 256 * 16 * 16 = 65,536
        flat_dim = 256 * 16 * 16
        
        # 5. 回归头
        self.mouse_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 2048),
            nn.LayerNorm(2048),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 512),
            nn.SiLU(),
            nn.Linear(512, 2) # dx, dy
        )
        
        self.click_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        feat = self.encoder(x)       # -> [B, 2048, 16, 16]
        feat = self.compressor(feat) # -> [B, 256, 16, 16]
        
        mouse_out = self.mouse_head(feat)
        click_out = self.click_head(feat)
        
        return mouse_out, click_out