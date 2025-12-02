import os
from PIL import Image
import matplotlib.pyplot as plt

# 分析训练数据
train_dir = 'g:/AndyL_osu/train_img'
categories = ['1', '2', '3']

print("=== 训练数据分布分析 ===")
for cat in categories:
    cat_dir = os.path.join(train_dir, cat)
    if os.path.exists(cat_dir):
        files = [f for f in os.listdir(cat_dir) if f.endswith('.png')]
        print(f'类别 {cat} 有 {len(files)} 张图像')
        print(f'示例文件: {files[:3]}')
        
        # 显示类别3的前5张图像（spinner）
        if cat == '3':
            print("\n=== Spinner 类别图像分析 ===")
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            for i, file in enumerate(files[:5]):
                img_path = os.path.join(cat_dir, file)
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].set_title(f'Spinner {i+1}')
                axes[i].axis('off')
            plt.tight_layout()
            plt.savefig('spinner_samples.png')
            print("Spinner 样本图像已保存为 spinner_samples.png")

# 检查spinner图像与其他类别的差异
print("\n=== 类别特征对比 ===")
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for row, cat in enumerate(categories):
    cat_dir = os.path.join(train_dir, cat)
    files = [f for f in os.listdir(cat_dir) if f.endswith('.png')]
    for col in range(3):
        if col < len(files):
            img_path = os.path.join(cat_dir, files[col])
            img = Image.open(img_path)
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'类别 {cat}')
            axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('category_comparison.png')
print("类别对比图像已保存为 category_comparison.png")
print("分析完成！")