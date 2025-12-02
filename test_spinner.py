import os
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn

# 确保设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 测试图像路径
TEST_IMG_PATH = 'test_img'
TRAIN_IMG_PATH = 'train_img'

# 直接定义所需的配置
CONFIG = {
    'num_classes': 4,
    'class_names': ['circle', 'slider', 'spinner', 'back'],
    'image_size': (160, 90),
    'batch_size': 128,
    'num_epochs': 50
}

# 直接定义OsuNet模型类，与main.py中的完全一致
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

# 加载模型
model = OsuNet(num_classes=CONFIG['num_classes'])
model_path = 'models/osu_model.pth'

if os.path.exists(model_path):
    try:
        print(f"加载模型: {model_path}")
        # 加载模型权重，处理类别数量不匹配的情况
        checkpoint = torch.load(model_path, map_location=device)
        model_dict = model.state_dict()
        
        # 过滤掉不匹配的权重（主要是分类器的最后一层）
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        # 更新模型权重
        model_dict.update(filtered_checkpoint)
        model.load_state_dict(model_dict, strict=False)
        
        model.to(device)
        model.eval()
        print("模型加载成功！")
        print(f"加载了 {len(filtered_checkpoint)} 个匹配的权重，跳过了 {len(checkpoint) - len(filtered_checkpoint)} 个不匹配的权重")
    except Exception as e:
        print(f"加载模型失败: {e}")
        exit()
else:
    print("模型文件不存在，请先训练模型！")
    exit()

# 预处理图像函数
def preprocess_image(image_path):
    """预处理单张图像以供模型输入"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(CONFIG['image_size'])
    image = np.array(image)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    # 应用与训练时相同的Normalize变换
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean[:, None, None]) / std[:, None, None]
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0)  # 添加batch维度
    return image.to(device)

# 测试模型函数
def test_model(image_path, true_label=None):
    """测试模型在单张图像上的表现"""
    # 预处理图像
    input_image = preprocess_image(image_path)
    
    # 模型预测
    with torch.no_grad():
        outputs = model(input_image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        pred_label = torch.argmax(probabilities).item()
        pred_prob = probabilities[0][pred_label].item()
    
    return pred_label, pred_prob

# 评估模型性能，计算详细指标
def evaluate_model_performance():
    """评估模型性能，计算准确率、召回率、F1分数等详细指标"""
    # 获取所有类别的图像
    categories = [1, 2, 3, 4]  # 类别目录
    category_labels = [0, 1, 2, 3]  # 模型预测的标签
    category_names = CONFIG['class_names']
    
    # 初始化混淆矩阵
    num_classes = len(category_names)
    confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    
    # 收集所有预测结果
    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = []
    image_paths = []
    
    print("\n===== 开始评估模型性能 =====")
    
    # 遍历所有类别目录
    for cat_dir in categories:
        true_label = cat_dir - 1  # 目录1->circle(0), 2->slider(1), 3->spinner(2)
        cat_path = os.path.join(TRAIN_IMG_PATH, str(cat_dir))
        
        if not os.path.exists(cat_path):
            continue
        
        # 获取该类别下的所有图像
        images = [os.path.join(cat_path, f) for f in os.listdir(cat_path) if f.endswith('.png')]
        
        for img_path in images:
            pred_label, pred_prob = test_model(img_path)
            
            # 更新混淆矩阵
            confusion_matrix[true_label][pred_label] += 1
            
            # 收集结果
            all_true_labels.append(true_label)
            all_pred_labels.append(pred_label)
            all_pred_probs.append(pred_prob)
            image_paths.append(img_path)
            
            # 打印预测结果
            true_name = category_names[true_label]
            pred_name = category_names[pred_label]
            if pred_label == true_label:
                result = "✅ 正确"
            else:
                result = "❌ 错误"
            print(f"{img_path}: 真实类别={true_name}, 预测类别={pred_name} ({pred_prob:.4f}) - {result}")
    
    # 计算详细指标
    print("\n===== 混淆矩阵 =====")
    print(f"{'':<12}", end="")
    for name in category_names:
        print(f"{name:<12}", end="")
    print()
    
    for i in range(num_classes):
        print(f"{category_names[i]:<12}", end="")
        for j in range(num_classes):
            print(f"{confusion_matrix[i][j]:<12}", end="")
        print()
    
    # 计算每个类别的指标
    print("\n===== 类别级指标 =====")
    precision = []
    recall = []
    f1 = []
    accuracy = []
    
    for i in range(num_classes):
        # 真阳性
        tp = confusion_matrix[i][i]
        # 假阳性（其他类别预测为当前类别）
        fp = sum(confusion_matrix[j][i] for j in range(num_classes) if j != i)
        # 假阴性（当前类别预测为其他类别）
        fn = sum(confusion_matrix[i][j] for j in range(num_classes) if j != i)
        # 真阴性
        tn = sum(confusion_matrix[j][k] for j in range(num_classes) for k in range(num_classes) if j != i and k != i)
        
        # 计算指标
        class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
        class_accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        
        precision.append(class_precision)
        recall.append(class_recall)
        f1.append(class_f1)
        accuracy.append(class_accuracy)
        
        print(f"类别: {category_names[i]}")
        print(f"  准确率: {class_accuracy:.2%}")
        print(f"  精确率: {class_precision:.2%}")
        print(f"  召回率: {class_recall:.2%}")
        print(f"  F1分数: {class_f1:.2%}")
    
    # 计算宏平均和微平均
    macro_precision = sum(precision) / num_classes
    macro_recall = sum(recall) / num_classes
    macro_f1 = sum(f1) / num_classes
    
    # 微平均（基于总TP、FP、FN）
    total_tp = sum(confusion_matrix[i][i] for i in range(num_classes))
    total_fp = sum(confusion_matrix[j][i] for i in range(num_classes) for j in range(num_classes) if j != i)
    total_fn = sum(confusion_matrix[i][j] for i in range(num_classes) for j in range(num_classes) if j != i)
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # 总体准确率
    overall_accuracy = total_tp / len(all_true_labels) if len(all_true_labels) > 0 else 0
    
    print("\n===== 总体指标 =====")
    print(f"总体准确率: {overall_accuracy:.2%}")
    print(f"宏平均精确率: {macro_precision:.2%}")
    print(f"宏平均召回率: {macro_recall:.2%}")
    print(f"宏平均F1分数: {macro_f1:.2%}")
    print(f"微平均精确率: {micro_precision:.2%}")
    print(f"微平均召回率: {micro_recall:.2%}")
    print(f"微平均F1分数: {micro_f1:.2%}")
    
    # 识别表现最差的类别
    worst_class_idx = f1.index(min(f1))
    print(f"\n===== 表现最差的类别 =====")
    print(f"类别: {category_names[worst_class_idx]}")
    print(f"F1分数: {f1[worst_class_idx]:.2%}")
    
    # 识别常见的错误预测
    print("\n===== 常见错误预测 =====")
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion_matrix[i][j] > 0:
                print(f"{category_names[i]} 被误判为 {category_names[j]}: {confusion_matrix[i][j]} 次")
    
    # 返回评估结果
    evaluation_results = {
        'confusion_matrix': confusion_matrix,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'overall_accuracy': overall_accuracy,
        'worst_class': category_names[worst_class_idx],
        'worst_class_f1': f1[worst_class_idx],
        'all_true_labels': all_true_labels,
        'all_pred_labels': all_pred_labels,
        'all_pred_probs': all_pred_probs,
        'image_paths': image_paths
    }
    
    return evaluation_results

# 可视化预测结果
def visualize_predictions():
    """可视化模型对不同类别的预测结果"""
    # 获取每个类别的示例图像
    categories = [1, 2, 3]
    category_names = CONFIG['class_names']
    
    fig, axes = plt.subplots(len(categories), 4, figsize=(15, 15))
    
    for i, cat in enumerate(categories):
        cat_dir = os.path.join(TRAIN_IMG_PATH, str(cat))
        if not os.path.exists(cat_dir):
            continue
        
        images = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.endswith('.png')][:4]
        
        for j, img_path in enumerate(images):
            # 读取图像
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 模型预测
            pred_label, pred_prob = test_model(img_path)
            
            # 设置标题
            # 类别目录: 1->circle(索引0), 2->slider(索引1), 3->spinner(索引2)
            true_idx = cat - 1
            true_name = category_names[true_idx]
            pred_name = category_names[pred_label]
            
            if pred_name == true_name:
                title_color = 'green'
            else:
                title_color = 'red'
            
            axes[i, j].imshow(img)
            axes[i, j].set_title(f"真实: {true_name}\n预测: {pred_name} ({pred_prob:.4f})", 
                              color=title_color, fontsize=12)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    print("\n📊 预测结果可视化已保存为 prediction_visualization.png")

# 主函数
if __name__ == "__main__":
    # 评估模型性能，计算详细指标
    evaluation_results = evaluate_model_performance()
    
    # 可视化预测结果
    visualize_predictions()
    
    # 生成改进建议
    print("\n===== 模型改进建议 =====")
    print("1. 数据增强策略：")
    print("   - 对spinner类别进行更多的数据增强，包括旋转、缩放、亮度调整等")
    print("   - 理由：spinner类别的样本数量较少，且与slider类别特征相似，需要更多多样化样本")
    print("   - 预期效果：提高spinner类别的召回率和F1分数，减少与slider的混淆")
    
    print("\n2. 特征工程优化：")
    print("   - 增加spinner特有的特征提取，如旋转线条、内外圆结构等")
    print("   - 理由：当前模型可能没有充分捕捉spinner的独特特征")
    print("   - 预期效果：增强模型对spinner的识别能力")
    
    print("\n3. 模型结构调整：")
    print("   - 增加模型的深度或宽度，特别是在分类头部分")
    print("   - 理由：当前模型可能容量不足，无法区分相似的类别")
    print("   - 预期效果：提高模型的分类能力，减少误判")
    
    print("\n4. 超参數調優：")
    print("   - 调整学习率调度策略，尝试更大的初始学习率")
    print("   - 增加训练轮数，让模型有更多时间学习spinner特征")
    print("   - 理由：当前训练轮数可能不足，模型尚未充分收敛")
    print("   - 预期效果：提高模型的整体性能")
    
    print("\n5. 类别权重调整：")
    print("   - 进一步增加spinner类别的权重，如从1.5调整到2.0")
    print("   - 理由：spinner类别表现最差，需要更多关注")
    print("   - 预期效果：提高spinner类别的召回率")
    
    print("\n测试完成！")
