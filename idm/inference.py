import torch
import cv2
import numpy as np
import argparse
import random
import os
import matplotlib.pyplot as plt
from collections import OrderedDict

# 引入你的项目配置
from config import Config
from model import OsuIDM

def load_model(checkpoint_path, device):
    """加载模型权重，自动处理 DDP 的 'module.' 前缀"""
    print(f"🔄 Loading model from {checkpoint_path}...")
    
    # 初始化模型结构
    model = OsuIDM().to(device)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 兼容处理：检查是存的整个 checkpoint 字典还是只有 state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # 去除 DDP 训练产生的 'module.' 前缀
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') # remove `module.`
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def preprocess_frames(frames):
    """
    将读取的 OpenCV 帧列表转换为模型输入的 Tensor
    逻辑需严格对齐 train.py / preprocess.py
    """
    processed_frames = []
    
    for frame in frames:
        # 1. Resize (Config.IMG_SIZE = 224)
        frame_resized = cv2.resize(frame, (Config.IMG_SIZE, Config.IMG_SIZE))
        
        # 2. 灰度化
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        processed_frames.append(frame_gray)
    
    # 3. 堆叠 (Sequence Length = 6) -> Shape: (6, 224, 224)
    frame_stack = np.array(processed_frames)
    
    # 4. 归一化 (0-255 -> 0.0-1.0) 对应 train.py 中的 float() / 255.0
    tensor = torch.from_numpy(frame_stack).float() / 255.0
    
    # 5. 增加 Batch 维度 -> Shape: (1, 6, 224, 224)
    tensor = tensor.unsqueeze(0)
    
    return tensor, frame_stack

def visualize_result(frame_stack, pred_mouse, pred_click, save_path="result.jpg"):
    """
    可视化逻辑：
    1. 6帧叠加 (每帧 1/6 透明度)
    2. 画箭头 (移动)
    3. 画红点 (点击)
    """
    H, W = frame_stack.shape[1], frame_stack.shape[2] # 224, 224
    
    # --- 1. 制作叠加背景 ---
    # 计算平均值来实现 1/6 透明度叠加效果
    composite_gray = np.mean(frame_stack, axis=0).astype(np.uint8)
    # 转回 BGR 以便画彩色线
    vis_img = cv2.cvtColor(composite_gray, cv2.COLOR_GRAY2BGR)
    
    # --- 2. 解析预测结果 ---
    # pred_mouse 是归一化的 dx, dy (基于 512x384)
    # 我们需要将其缩放到当前图片尺寸 (224x224) 以便可视化
    dx_norm, dy_norm = pred_mouse[0], pred_mouse[1]
    click_prob = pred_click
    
    # 还原到原始 osu 坐标系的位移 (512x384)
    real_dx = dx_norm * 512.0
    real_dy = dy_norm * 384.0
    
    # 映射到可视化图片的尺寸 (这里为了显示明显，稍微放大一点比例，或者直接按比例映射)
    # 图片宽 224，osu 宽 512 -> 比例约 0.43
    scale_x = W / 512.0
    scale_y = H / 384.0
    
    vis_dx = int(real_dx * scale_x * 5.0) # *5 是为了让微小的移动在图上肉眼更明显
    vis_dy = int(real_dy * scale_y * 5.0)
    
    # 设定中心点
    center_x, center_y = W // 2, H // 2
    end_x, end_y = center_x + vis_dx, center_y + vis_dy
    
    # --- 3. 绘制 ---
    
    # A. 绘制移动箭头 (绿色)
    # 提示：IDM 模型预测的是“光标的相对移动(Velocity)”，而不是绝对位置
    # 所以我们从画面中心画出这个向量
    cv2.arrowedLine(vis_img, (center_x, center_y), (end_x, end_y), (0, 255, 0), 2, tipLength=0.3)
    
    # B. 绘制点击状态
    is_clicking = click_prob > 0.5
    status_text = f"Click: {click_prob:.2f}"
    
    if is_clicking:
        # 如果点击，在中心画一个红色的实心圆
        cv2.circle(vis_img, (center_x, center_y), 10, (0, 0, 255), -1) 
        text_color = (0, 0, 255) # Red
    else:
        # 没点击，画一个空心蓝圆
        cv2.circle(vis_img, (center_x, center_y), 10, (255, 0, 0), 1)
        text_color = (255, 0, 0) # Blue
        
    # C. 添加文字信息
    cv2.putText(vis_img, f"dx:{dx_norm:.3f} dy:{dy_norm:.3f}", (5, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(vis_img, status_text, (5, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # 保存
    cv2.imwrite(save_path, vis_img)
    print(f"✅ Result saved to {save_path}")
    print(f"   Pred: Move({dx_norm:.4f}, {dy_norm:.4f}), Click({click_prob:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Osu IDM Inference Tool")
    parser.add_argument("--video", type=str, required=True, help="Path to mp4 video")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 1. 准备模型
    model = load_model(args.checkpoint, args.device)

    # 2. 读取视频并随机抽样
    if not os.path.exists(args.video):
        print("❌ Video file not found.")
        return

    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 10:
        print("❌ Video too short.")
        return

    # 随机选择一个起始点 (确保后面有6帧)
    start_idx = random.randint(0, total_frames - 7)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    
    frames = []
    print(f"🎬 Sampling 6 frames starting from index {start_idx}...")
    for _ in range(6):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    
    if len(frames) != 6:
        print("❌ Failed to read 6 consecutive frames.")
        return

    # 3. 预处理
    input_tensor, frame_stack = preprocess_frames(frames)
    input_tensor = input_tensor.to(args.device)

    # 4. 推理
    with torch.no_grad():
        # 模型输出: mouse -> (B, 2), click -> (B, 1)
        pred_mouse, pred_click = model(input_tensor)
        
        # 获取数值
        dx_dy = pred_mouse[0].cpu().numpy() # [dx, dy]
        click_logit = pred_click[0].item()
        click_prob = torch.sigmoid(pred_click[0]).item()

    # 5. 可视化
    visualize_result(frame_stack, dx_dy, click_prob, save_path=f"inference_frame_{start_idx}.jpg")

if __name__ == "__main__":
    main()