import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import win32api as api32
from mss import mss
from torchvision import transforms
import keyboard as kb
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 简单的图像变换，确保返回正确的张量类型
def simple_transform(image_size=(160, 90)):
    def transform_fn(img):
        # 确保输入是numpy数组
        if not isinstance(img, np.ndarray):
            print(f"警告: 输入不是numpy数组，类型: {type(img)}")
            return torch.zeros((3, image_size[1], image_size[0]), dtype=torch.float32)
        
        # 确保是 RGB 格式
        if len(img.shape) == 3 and img.shape[2] == 3:
            # 调整大小
            img = cv2.resize(img, image_size)
            # 转换为张量
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            # 归一化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            return img_tensor
        else:
            # 如果不是3通道图像，转换为灰度然后再转RGB
            img = cv2.resize(img, image_size)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            # 归一化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            return img_tensor
    return transform_fn

# 默认变换
default_transform = simple_transform()

class OsuNet(nn.Module):
    def __init__(self, num_classes=4, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # 1. 骨干网络（特征提取）
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # [B, 16, 80, 45]
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # [B, 32, 40, 22]
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # [B, 64, 20, 11]
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # [B, 128, 10, 5]
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),  # [B, 256, 10, 5]
        )
        
        # 2. 检测头
        pred_per_anchor = 5 + num_classes
        self.detection_head = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(256, num_anchors * pred_per_anchor, kernel_size=1)
        )
        
        # 3. 锚框设置
        self.register_buffer('anchors', torch.tensor([
            [0.08, 0.08],  # 小圆圈
            [0.15, 0.15],  # 中滑条头部
            [0.25, 0.10],  # 长滑条
        ], dtype=torch.float32))
        
        # 4. 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        predictions = self.detection_head(features)
        return predictions
    
    def decode_predictions(self, predictions, conf_thresh=0.5, nms_thresh=0.4):
        batch_size = predictions.shape[0]
        grid_h, grid_w = 5, 10
        
        # 尝试正确地重塑张量
        try:
            pred = predictions.view(
                batch_size, self.num_anchors, -1, grid_h, grid_w
            ).permute(0, 3, 4, 1, 2).contiguous()
        except RuntimeError as e:
            # 如果重塑失败，打印错误信息并尝试不同的方法
            print(f"张量重塑错误: {e}")
            print(f"原始形状: {predictions.shape}")
            
            # 如果原始张量是 [B, H, W, C] 格式，我们需要先将其重塑为 [B, C, H, W]
            if len(predictions.shape) == 4 and predictions.shape[3] > predictions.shape[1]:
                # 假设形状是 [B, H, W, C]
                predictions = predictions.permute(0, 3, 1, 2).contiguous()
            
            # 尝试使用新的形状
            try:
                pred = predictions.view(
                    batch_size, self.num_anchors, -1, grid_h, grid_w
                ).permute(0, 3, 4, 1, 2).contiguous()
            except RuntimeError as e2:
                print(f"重试失败: {e2}")
                # 返回空结果，避免崩溃
                return {
                    'boxes': torch.empty((0, 4)),
                    'scores': torch.empty((0)),
                    'classes': torch.empty((0), dtype=torch.long)
                }
        
        output = []
        
        for b in range(batch_size):
            boxes = []
            scores = []
            class_ids = []
            
            for y in range(grid_h):
                for x in range(grid_w):
                    for a in range(self.num_anchors):
                        pred_vec = pred[b, y, x, a]
                        
                        tx, ty, tw, th, obj_conf = pred_vec[:5].sigmoid()
                        cls_scores = pred_vec[5:].softmax(dim=0)
                        cls_score, class_id = torch.max(cls_scores, dim=0)
                        
                        final_score = obj_conf * cls_score
                        if final_score < conf_thresh:
                            continue
                        
                        cx = (x + tx) / grid_w * 160
                        cy = (y + ty) / grid_h * 90
                        w = self.anchors[a][0] * torch.exp(tw) * 160
                        h = self.anchors[a][1] * torch.exp(th) * 90
                        
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2
                        
                        boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                        scores.append(final_score.item())
                        class_ids.append(class_id.item())
            
            if boxes:
                boxes_tensor = torch.tensor(boxes)
                scores_tensor = torch.tensor(scores)
                class_tensor = torch.tensor(class_ids, dtype=torch.long)
                
                keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, nms_thresh)
                
                if len(keep_indices) > 0:
                    filtered_boxes = boxes_tensor[keep_indices]
                    filtered_scores = scores_tensor[keep_indices]
                    filtered_classes = class_tensor[keep_indices]
                    
                    output.append({
                        'boxes': filtered_boxes,
                        'scores': filtered_scores,
                        'classes': filtered_classes
                    })
                else:
                    output.append({
                        'boxes': torch.empty((0, 4)),
                        'scores': torch.empty((0)),
                        'classes': torch.empty((0), dtype=torch.long)
                    })
            else:
                output.append({
                    'boxes': torch.empty((0, 4)),
                    'scores': torch.empty((0)),
                    'classes': torch.empty((0), dtype=torch.long)
                })
        
        # 将批次结果合并为一个字典
        if output:
            return {
                'boxes': torch.cat([r['boxes'] for r in output]),
                'scores': torch.cat([r['scores'] for r in output]),
                'classes': torch.cat([r['classes'] for r in output])
            }
        else:
            return {
                'boxes': torch.empty((0, 4)),
                'scores': torch.empty((0)),
                'classes': torch.empty((0), dtype=torch.long)
            }


class YOLOLoss(nn.Module):
    """修复后的YOLO损失函数"""
    def __init__(self, num_classes, num_anchors, anchors, img_size=(160, 90), device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.anchors = anchors  # 传入模型的锚框
        self.img_size = img_size
        self.device = device
        
        # 损失权重
        self.lambda_coord = 5.0
        self.lambda_obj = 1.0
        self.lambda_noobj = 0.5
        self.lambda_cls = 1.0
        
        # 网格尺寸
        self.grid_h, self.grid_w = 5, 10
        
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        
    def forward(self, predictions, targets):
        batch_size = predictions.shape[0]
        
        pred = predictions.view(
            batch_size, self.num_anchors, -1, self.grid_h, self.grid_w
        ).permute(0, 3, 4, 1, 2).contiguous()
        
        loss_coord = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_obj = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_noobj = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_cls = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for b in range(batch_size):
            target_boxes = targets[b]
            if len(target_boxes) == 0:
                continue
            
            for target_idx in range(len(target_boxes)):
                class_id, tx, ty, tw, th = target_boxes[target_idx]
                
                grid_x = int(tx * self.grid_w)
                grid_y = int(ty * self.grid_h)
                
                if grid_x >= self.grid_w or grid_y >= self.grid_h:
                    continue
                
                # 计算与所有锚框的IoU
                best_iou = 0
                best_anchor = 0
                
                for a in range(self.num_anchors):
                    anchor_w, anchor_h = self.anchors[a]
                    inter = min(tw, anchor_w) * min(th, anchor_h)
                    union = tw * th + anchor_w * anchor_h - inter
                    iou = inter / (union + 1e-6)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_anchor = a
                
                if best_iou < 0.3:
                    continue
                
                # 坐标损失
                pred_box = pred[b, grid_y, grid_x, best_anchor, :4]
                tx_pred, ty_pred, tw_pred, th_pred = pred_box
                
                tx_true = tx * self.grid_w - grid_x
                ty_true = ty * self.grid_h - grid_y
                tw_true = torch.log(tw / self.anchors[best_anchor][0] + 1e-6)
                th_true = torch.log(th / self.anchors[best_anchor][1] + 1e-6)
                
                loss_coord = loss_coord + self.mse_loss(tx_pred, tx_true)
                loss_coord = loss_coord + self.mse_loss(ty_pred, ty_true)
                loss_coord = loss_coord + self.mse_loss(tw_pred, tw_true)
                loss_coord = loss_coord + self.mse_loss(th_pred, th_true)
                
                # 物体置信度损失
                pred_obj = pred[b, grid_y, grid_x, best_anchor, 4]
                loss_obj = loss_obj + self.bce_loss(pred_obj.sigmoid(), torch.tensor(1.0, device=self.device))
                
                # 分类损失
                pred_cls = pred[b, grid_y, grid_x, best_anchor, 5:]
                class_tensor = torch.tensor(class_id, dtype=torch.long, device=self.device).unsqueeze(0)
                loss_cls = loss_cls + self.ce_loss(pred_cls.unsqueeze(0), class_tensor)
                
                # 负样本损失
                for gy in range(self.grid_h):
                    for gx in range(self.grid_w):
                        for a in range(self.num_anchors):
                            if not (gy == grid_y and gx == grid_x and a == best_anchor):
                                pred_obj_neg = pred[b, gy, gx, a, 4]
                                loss_noobj = loss_noobj + self.lambda_noobj * self.bce_loss(
                                    pred_obj_neg.sigmoid(), 
                                    torch.tensor(0.0, device=self.device)
                                )
        
        # 归一化损失
        total_boxes = sum(len(t) for t in targets)
        if total_boxes > 0:
            loss_coord = loss_coord / total_boxes
            loss_cls = loss_cls / total_boxes
        
        total_loss = (self.lambda_coord * loss_coord + 
                     self.lambda_obj * loss_obj + 
                     self.lambda_noobj * loss_noobj + 
                     self.lambda_cls * loss_cls)
        
        # 确保total_loss是tensor类型
        if not isinstance(total_loss, torch.Tensor):
            total_loss = torch.tensor(total_loss, device=self.device, requires_grad=True)
        
        return total_loss, (loss_coord, loss_obj, loss_noobj, loss_cls)


class OsuDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, img_size=(160, 90), num_classes=4):
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size
        self.num_classes = num_classes
        self.image_files = []
        self.annotations = []
        
        self._load_data()
    
    def _load_data(self):
        """修复后的数据加载函数"""
        # 检查数据目录是否存在
        if not os.path.exists(self.data_dir):
            print(f"错误: 数据目录不存在: {self.data_dir}")
            return
        
        # 尝试不同的目录结构
        
        images_dir = None
        labels_dir = None
        
        if os.path.exists(self.data_dir):
            # 检查是否有images和labels子目录
            images = os.path.join(self.data_dir, 'images')
            labels = os.path.join(self.data_dir, 'labels')
            
            if os.path.exists(images) and os.path.exists(labels):
                images_dir = images
                labels_dir = labels
        
        print(f"使用图像目录: {images_dir}")
        print(f"使用标注目录: {labels_dir}")
        
        loaded_count = 0
        skipped_count = 0
        
        # 遍历图像目录
        for root, dirs, files in os.walk(images_dir):
            for img_file in files:
                if img_file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(root, img_file)
                    
                    # 构建对应的标注文件路径
                    rel_path = os.path.relpath(root, images_dir)
                    label_root = os.path.join(labels_dir, rel_path) if labels_dir != images_dir else root
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    label_path = os.path.join(label_root, label_file)
                    
                    if os.path.exists(label_path):
                        self.image_files.append(img_path)
                        
                        # 加载标注
                        boxes = []
                        try:
                            with open(label_path, 'r', encoding='utf-8') as f:
                                for line_num, line in enumerate(f, 1):
                                    line = line.strip()
                                    if not line or line.startswith('#'):
                                        continue
                                    
                                    parts = line.split()
                                    if len(parts) >= 5:
                                        try:
                                            class_id = int(parts[0])
                                            x_center = float(parts[1])
                                            y_center = float(parts[2])
                                            width = float(parts[3])
                                            height = float(parts[4])
                                            
                                            # 验证坐标
                                            if (0 <= class_id < self.num_classes and 
                                                0 <= x_center <= 1 and 0 <= y_center <= 1 and
                                                0 <= width <= 1 and 0 <= height <= 1):
                                                boxes.append([class_id, x_center, y_center, width, height])
                                            else:
                                                print(f"警告: 坐标超出范围 {label_path}:{line_num}")
                                        except ValueError as e:
                                            print(f"警告: 格式错误 {label_path}:{line_num} - {e}")
                                            continue
                        except Exception as e:
                            print(f"警告: 无法读取 {label_path}: {e}")
                            boxes = []
                        
                        self.annotations.append(boxes)
                        loaded_count += 1
                    else:
                        skipped_count += 1
        
        print(f"数据集加载完成:")
        print(f"  - 成功加载: {loaded_count} 个图像文件")
        print(f"  - 跳过/缺失: {skipped_count} 个文件")
        print(f"  - 总计: {len(self.image_files)} 个有效样本")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        boxes = self.annotations[idx]
        
        try:
            # 加载图像
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"无法加载图像: {img_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            
            # 应用变换
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                # 归一化
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = (img - mean) / std
            
            # 标注
            if boxes:
                target = torch.tensor(boxes, dtype=torch.float32)
                cls_label = int(boxes[0][0])
            else:
                target = torch.empty((0, 5), dtype=torch.float32)
                cls_label = 0
            
            # 返回正确的格式：单个图像、单个标签、单个目标
            return img, cls_label, target
            
        except Exception as e:
            print(f"警告: 处理图像 {img_path} 时出错: {e}")
            default_img = torch.zeros((3, self.img_size[1], self.img_size[0]), dtype=torch.float32)
            return default_img, 0, torch.empty((0, 5), dtype=torch.float32)


def train_detection_model(model, train_loader, val_loader, epochs=50, device='cuda'):
    """修复后的训练函数"""
    model.to(device)
    
    # 损失函数 - 传入模型的锚框
    loss_fn = YOLOLoss(
        num_classes=model.num_classes,
        num_anchors=model.num_anchors,
        anchors=model.anchors,  # 传入锚框
        img_size=(160, 90),
        device=device
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.001, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        loss_components = {'coord': 0.0, 'obj': 0.0, 'noobj': 0.0, 'cls': 0.0}
        
        # 训练阶段
        for batch_idx, (images, cls_labels, targets) in enumerate(train_loader):
            images = images.to(device)
            
            # 只需要检测目标
            target_list = []
            for t in targets:
                target_list.append(t.to(device) if len(t) > 0 else torch.empty((0, 5), device=device))
            
            # 前向传播
            predictions = model(images)
            
            # 计算损失
            loss, (loss_c, loss_o, loss_no, loss_cls) = loss_fn(predictions, target_list)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # 记录损失
            train_loss += loss.item()
            loss_components['coord'] += loss_c.item()
            loss_components['obj'] += loss_o.item()
            loss_components['noobj'] += loss_no.item()
            loss_components['cls'] += loss_cls.item()
            
            # 进度显示
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs} | '
                      f'Batch {batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | '
                      f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # 计算平均损失
        avg_loss = train_loss / len(train_loader)
        for key in loss_components:
            loss_components[key] /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, cls_labels, targets in val_loader:
                images = images.to(device)
                target_list = [t.to(device) if len(t) > 0 else torch.empty((0, 5), device=device) for t in targets]
                
                predictions = model(images)
                loss, _ = loss_fn(predictions, target_list)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'\nEpoch {epoch+1}/{epochs} Summary:')
        print(f'  Train Loss: {avg_loss:.4f}')
        print(f'    Coord: {loss_components["coord"]:.4f}, '
              f'Obj: {loss_components["obj"]:.4f}, '
              f'NoObj: {loss_components["noobj"]:.4f}, '
              f'Cls: {loss_components["cls"]:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 50)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_osu_detector.pth')
            print(f'  ✅ 保存最佳模型 (loss: {best_loss:.4f})')
    
    return model

def get_aicam():
    with mss() as sct:
        monitor = sct.monitors[1]
        img = sct.grab(monitor)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        img[img <= 180] = 0
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

def play_game(model: OsuNet):
    running = False
    success = False
    silder_hold = False
    spin_hold = False
    spin_center_x = 0
    spin_center_y = 0
    spin_angle = 0
    key_down_time = None
    press_duration_threshold = 0.5  # 长按阈值（秒）
    
    def on_key_event(event):
        nonlocal running, success, silder_hold, spin_hold, key_down_time
        
        if event.name == 'q':
            if event.event_type == 'down':
                # 记录按键按下时间
                key_down_time = time.time()
                
                if not running:
                    running = True
                    print("游戏模式启动 - 按 q 暂停")
                else:
                    running = False
                    print("游戏模式暂停 - 按 q 继续")
                    # 释放所有按键状态
                    silder_hold = False
                    spin_hold = False
            elif event.event_type == 'up' and running:
                # 计算按键持续时间
                if key_down_time is not None:
                    press_duration = time.time() - key_down_time
                    
                    # 如果持续时间超过阈值，则为长按
                    if press_duration >= press_duration_threshold:
                        success = True
                        print("游戏模式退出")
                        return False
                    
                    # 重置按键按下时间
                    key_down_time = None
    
    kb.on_press(on_key_event)
    
    print("Osu AI 游戏助手启动")
    print("按 q 开始/暂停游戏")
    print("按住 q 退出")
    
    # 帧率控制
    import time
    target_fps = 30
    frame_time = 1.0 / target_fps
    
    model.eval()
    with torch.no_grad():
        # 确保模型和输入在相同设备上
        model = model.to(device)
        
        while not success:
            if not running:
                time.sleep(0.1)
                continue
                
            start_time = time.time()
            
            try:
                # 获取屏幕截图
                image = get_aicam()
                if image is None or image.size == 0:
                    print("无法获取屏幕图像")
                    continue
                    
                # 模型预测
                image_tensor = image.astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1).unsqueeze(0).to(device)
                
                predictions = model(image_tensor)
                results = model.decode_predictions(predictions)
                
                # 打印检测结果（调试用）
                if results['scores'].shape[0] > 0:
                    print(f"检测到 {results['scores'].shape[0]} 个目标")
                    for i in range(min(3, results['scores'].shape[0])):  # 最多显示3个
                        score = results['scores'][i].item()
                        class_id = results['classes'][i].item()
                        print(f"  目标 {i+1}: 类别={class_id}, 置信度={score:.3f}")
                
                # 执行游戏操作
                if results['scores'].shape[0] > 0:
                    # 检查置信度阈值
                    valid_detections = results['scores'] > 0.7
                    if torch.any(valid_detections):
                        # 获取最高置信度的检测结果
                        best_idx = torch.argmax(results['scores'])
                        best_class = results['classes'][best_idx].item()
                        best_box = results['boxes'][best_idx].cpu().numpy()
                        
                        # 计算点击位置（边界框中心）
                        if len(best_box) >= 4:
                            center_x = int((best_box[0] + best_box[2]) / 2)
                            center_y = int((best_box[1] + best_box[3]) / 2)
                            
                            # 执行相应的鼠标操作
                            if best_class == 0:
                                # 點擊圓圈
                                api32.SetCursorPos((center_x, center_y))
                                api32.mouse_event(0x0002, center_x, center_y, 0, 0)  # MOUSEEVENTF_LEFTDOWN
                                api32.mouse_event(0x0004, center_x, center_y, 0, 0)  # MOUSEEVENTF_LEFTUP
                                print(f"執行點擊圓圈: ({center_x}, {center_y})")
                            elif best_class == 1:
                                # 按住滑塊
                                api32.SetCursorPos((center_x, center_y))
                                if not silder_hold:
                                    api32.mouse_event(0x0002, center_x, center_y, 0, 0)  # MOUSEEVENTF_LEFTDOWN
                                    silder_hold = True
                                    print(f"按住滑塊: ({center_x}, {center_y})")
                            elif best_class == 2:
                                # 旋轉(spin)操作
                                if silder_hold:
                                    silder_hold = False
                                    api32.mouse_event(0x0004, center_x, center_y, 0, 0)  # MOUSEEVENTF_LEFTUP
                                
                                if not spin_hold:
                                    # 開始旋轉
                                    spin_hold = True
                                    spin_center_x = center_x
                                    spin_center_y = center_y
                                    spin_angle = 0
                                    api32.mouse_event(0x0002, center_x, center_y, 0, 0)  # MOUSEEVENTF_LEFTDOWN
                                    print(f"開始旋轉: 中心({center_x}, {center_y})")
                                
                                # 持續旋轉邏輯
                                if spin_hold:
                                    # 旋轉參數
                                    radius = 15  # 旋轉半徑
                                    angular_speed = 90  # 角度/秒
                                    spin_speed = 1.0 / target_fps  # 每幀時間
                                    
                                    # 更新角度
                                    spin_angle += angular_speed * spin_speed
                                    if spin_angle >= 360:
                                        spin_angle -= 360
                                    
                                    # 計算新位置
                                    radian = np.radians(spin_angle)
                                    x = int(spin_center_x + radius * np.cos(radian))
                                    y = int(spin_center_y + radius * np.sin(radian))
                                    
                                    # 移動鼠標到新位置
                                    api32.SetCursorPos((x, y))
                            
                            elif best_class == 3:
                                # 類別3的處理 這個是滑條返回的遊戲提示所以不用做動作
                                pass
                            else:
                                spin_hold = False
                                silder_hold = False
                                mouse_event(0x0004, center_x, center_y, 0, 0)  # MOUSEEVENTF_LEFTUP                 
                else:
                    spin_hold = False
                    silder_hold = False
                    mouse_event(0x0004, center_x, center_y, 0, 0)  # MOUSEEVENTF_LEFTUP                                     
            except Exception as e:
                print(f"遊戲循環錯誤: {e}")
                continue
            
            # 帧率控制
            elapsed_time = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    print("游戏模式已退出")
    kb.unhook_all()
            

def main():
    """主函数"""
    # 配置参数
    data_dir = "./train_img"  # 修改为你的数据目录
    batch_size = 16
    epochs = 50
    
    # 创建数据集
    print("加载数据集...")
    dataset = OsuDetectionDataset(
        data_dir=data_dir,
        transform=default_transform,
        img_size=(160, 90),
        num_classes=4
    )
    
    if len(dataset) == 0:
        print("错误: 未找到训练数据！请检查数据目录结构。")
        print("期望的结构:")
        print(f"  {data_dir}/")
        print("    ├── images/")
        print("    │   └── *.jpg (或 *.png)")
        print("    └── labels/")
        print("        └── *.txt (YOLO格式标注)")
        return
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    def collate_fn(batch):
        # batch是一个列表，每个元素是 (img, cls_label, target)
        # 需要转换为批次格式
        images = []
        cls_labels = []
        targets = []
        
        for img, cls_label, target in batch:
            images.append(img)
            cls_labels.append(cls_label)
            targets.append(target)
        
        # 堆叠图像张量
        images = torch.stack(images, dim=0)
        
        # 转换分类标签为张量
        cls_labels = torch.tensor(cls_labels, dtype=torch.long)
        
        return images, cls_labels, targets
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 创建模型
    print("创建模型...")
    model = OsuNet(num_classes=4, num_anchors=3)
    
    # 如果有预训练模型则加载
    model_path = 'best_osu_detector.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"加载预训练模型: {model_path}")
    
    # 训练模型
    if not os.path.exists(model_path):
        print("开始训练...")
        model = train_detection_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            device=device
        )
    else:
        user = input("在訓練多一次？(Y/n)：")
        if user == "Y" or user == "y" or not user:
            print("开始训练...")
            model = train_detection_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                device=device
            )
    
    # 保存最终模型
    final_model_path = 'osu_detector_final.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"训练完成！模型已保存到: {final_model_path}")

    play_game(model)

if __name__ == "__main__":
    main()