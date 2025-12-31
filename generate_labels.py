#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动标签生成脚本 - 从train_img目录检测轮廓并生成YOLO格式标签
"""

import cv2
import numpy as np
import os
from pathlib import Path

class AutoLabelGenerator:
    def __init__(self, train_img_dir, labels_dir=None):
        self.train_img_dir = Path(train_img_dir)
        self.labels_dir = Path(labels_dir) if labels_dir else self.train_img_dir.parent / "labels"
        
        # 创建labels目录
        self.labels_dir.mkdir(exist_ok=True)
        
        # 轮廓检测参数
        self.min_contour_area = 50  # 最小轮廓面积
        self.max_contour_area = 50000  # 最大轮廓面积
        self.white_threshold = 254  # 白色像素阈值（过滤背景）
        
        # 图像尺寸配置（需要与训练脚本保持一致）
        self.target_size = (160, 90)
        
        print(f"自动标签生成器初始化完成")
        print(f"  - 图像目录: {self.train_img_dir}")
        print(f"  - 标签目录: {self.labels_dir}")
        print(f"  - 目标尺寸: {self.target_size}")
        print(f"  - 轮廓面积范围: {self.min_contour_area} - {self.max_contour_area}")
        
    def is_white_background(self, image):
        """检查是否为白色背景图像"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 计算白色像素比例
        white_pixels = np.sum(gray >= self.white_threshold)
        total_pixels = gray.size
        white_ratio = white_pixels / total_pixels
        
        # 如果白色像素超过70%，认为是白色背景
        return white_ratio > 0.7
    
    def detect_contours(self, image):
        """检测图像中的轮廓"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 应用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    def filter_contours(self, contours):
        """过滤轮廓 - 去除白色背景和小轮廓"""
        filtered_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 面积过滤
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
                
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 宽高比过滤（避免过细的线段）
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 20:  # 长宽比超过20的过滤掉
                continue
                
            filtered_contours.append((contour, (x, y, w, h)))
        
        return filtered_contours
    
    def filter_contours_on_white_background(self, contours, image):
        """在白色背景上过滤轮廓 - 只保留非白色轮廓"""
        filtered_contours = []
        
        # 转换为HSV色彩空间更容易检测非白色
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义白色范围
        # HSV中白色的范围较宽，我们需要排除接近白色的像素
        lower_white = np.array([0, 0, 200])     # H:0-180, S:0-30, V:200-255
        upper_white = np.array([180, 30, 255])
        
        # 创建非白色掩码
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        non_white_mask = cv2.bitwise_not(white_mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 面积过滤
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
                
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 宽高比过滤（避免过细的线段）
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 20:  # 过滤掉过细的线段
                continue
            
            # 检查轮廓是否包含非白色像素
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # 计算轮廓区域内非白色像素的比例
            non_white_in_contour = cv2.bitwise_and(non_white_mask, mask)
            non_white_pixels = np.sum(non_white_in_contour > 0)
            total_pixels_in_contour = np.sum(mask > 0)
            
            if total_pixels_in_contour > 0:
                non_white_ratio = non_white_pixels / total_pixels_in_contour
                # 如果非白色像素占比超过30%，认为是有效轮廓
                if non_white_ratio > 0.3:
                    filtered_contours.append((contour, (x, y, w, h)))
        
        return filtered_contours
    
    def contours_to_yolo_format(self, contours_info, image_shape):
        """将轮廓转换为YOLO格式标签"""
        h, w = image_shape[:2]
        labels = []
        
        for contour, (x, y, bw, bh) in contours_info:
            # 计算中心点（相对于图像尺寸的比例）
            center_x = (x + bw / 2) / w
            center_y = (y + bh / 2) / h
            
            # 计算宽度和高度（相对于图像尺寸的比例）
            width = bw / w
            height = bh / h
            
            # 简单的类别分类（可以根据需要调整）
            area = bw * bh
            aspect_ratio = max(bw, bh) / min(bw, bh)
            
            # 基于大小和形状简单分类
            if area < 500:
                class_id = 0  # 小目标（如小圆圈）
            elif aspect_ratio < 1.5:
                class_id = 1  # 圆形或接近圆形
            elif aspect_ratio < 3.0:
                class_id = 2  # 椭圆或中等长宽比
            else:
                class_id = 3  # 长条形（可能为slider）
            
            # YOLO格式: class_id center_x center_y width height
            label = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
            labels.append(label)
        
        return labels
    
    def process_image(self, image_path):
        """处理单个图像文件"""
        try:
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"警告: 无法读取图像 {image_path}")
                return []
            
            original_shape = image.shape
            
            # 调整图像大小
            image = cv2.resize(image, self.target_size)
            
            # 检查是否为白色背景
            is_white_bg = self.is_white_background(image)
            
            # 检测轮廓
            contours = self.detect_contours(image)
            
            # 过滤轮廓
            if is_white_bg:
                # 对于白色背景图像，过滤掉白色的轮廓，只保留有颜色的轮廓
                filtered_contours = self.filter_contours_on_white_background(contours, image)
            else:
                # 对于非白色背景图像，正常过滤轮廓
                filtered_contours = self.filter_contours(contours)
            
            # 转换为YOLO格式
            yolo_labels = self.contours_to_yolo_format(filtered_contours, image.shape)
            
            if is_white_bg:
                print(f"检测到白色背景: {image_path.name}, 找到 {len(yolo_labels)} 个非白色目标")
            
            return yolo_labels
            
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            return []
    
    def process_directory(self):
        """处理整个目录"""
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        processed_count = 0
        total_labels = 0
        
        print("\n开始处理图像文件...")
        
        # 递归扫描所有子目录
        for image_file in self.train_img_dir.rglob("*"):
            if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                # 生成对应的标签文件路径
                relative_path = image_file.relative_to(self.train_img_dir)
                label_file = self.labels_dir / relative_path.with_suffix('.txt')
                
                # 创建子目录（如果需要）
                label_file.parent.mkdir(parents=True, exist_ok=True)
                
                print(f"处理: {relative_path}")
                
                # 处理图像并生成标签
                labels = self.process_image(image_file)
                
                # 保存标签文件
                if labels:
                    with open(label_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(labels))
                    print(f"  生成 {len(labels)} 个标签 -> {label_file.relative_to(self.labels_dir)}")
                    total_labels += len(labels)
                else:
                    # 即使没有标签也创建空文件（保持目录结构一致）
                    label_file.touch()
                    print(f"  无检测目标")
                
                processed_count += 1
        
        print(f"\n处理完成!")
        print(f"  - 处理图像: {processed_count} 个")
        print(f"  - 生成标签: {total_labels} 个")
        print(f"  - 标签文件保存在: {self.labels_dir}")
        
        return processed_count, total_labels

def main():
    """主函数"""
    print("=== Osu自动标签生成器 ===")
    
    # 设置路径
    train_img_dir = "train_img/images"
    labels_dir = "train_img/labels"
    
    # 检查图像目录是否存在
    if not os.path.exists(train_img_dir):
        print(f"错误: 图像目录 '{train_img_dir}' 不存在!")
        print("请确保 train_img 目录存在并包含图像文件。")
        return
    
    # 创建标签生成器
    generator = AutoLabelGenerator(train_img_dir, labels_dir)
    
    # 处理目录
    processed_count, total_labels = generator.process_directory()
    
    print("\n=== 完成 ===")
    if processed_count > 0:
        print(f"成功处理 {processed_count} 个图像文件")
        print(f"生成 {total_labels} 个目标标签")
        print(f"标签文件位置: {labels_dir}/")
        print("\n现在可以使用这些标签文件进行训练:")
        print(f"python main.py")
    else:
        print("没有找到可处理的图像文件")

if __name__ == "__main__":
    main()