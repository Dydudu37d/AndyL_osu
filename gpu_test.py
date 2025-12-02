import torch

def check_memory(batch_size=1024):
    # 模拟一个样本的大小
    sample_size = 224 * 224 * 3  # 假设RGB图像
    params_size = 10_000_000    # 假设1千万参数
    
    # 估算显存
    activation_memory = batch_size * sample_size * 4  # bytes
    gradient_memory = params_size * 4 * 3  # 参数梯度、动量等
    
    total_gb = (activation_memory + gradient_memory) / (1024**3)
    
    print(f"Batch size: {batch_size}")
    print(f"预估显存需求: {total_gb:.2f} GB")
    
    # 检查实际可用显存
    if torch.cuda.is_available():
        free_mem = torch.cuda.get_device_properties(0).total_memory / (int(input("请输入batch_size: "))**3)
        print(f"GPU显存: {free_mem:.2f} GB")
        
        if total_gb > free_mem * 0.8:  # 保留20%余量
            print("⚠️  batch_size太大！建议减小")
            print(f"当前显存: {free_mem:.2f} GB")
            print(f"建议显存需求: {total_gb * 0.8:.2f} GB")

check_memory()