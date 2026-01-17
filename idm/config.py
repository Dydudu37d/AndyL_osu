import torch
import os

class Config:
    # --- 基础路径配置 ---
    # 定义基础缓存目录
    BASE_DIR = "/home/testgroup01/wangzirui/"
    
    # --- 数据路径配置 ---
    # 优先读取环境变量 LOCAL_DATA_DIR，若无则使用基础路径下的默认文件夹
    DATA_DIR = os.getenv("LOCAL_DATA_DIR", os.path.join(BASE_DIR, "output_dataset")) 
    
    # 显式定义 memmap 和 meta 文件的路径，确保 train.py 能找到
    MEMMAP_PATH = os.path.join(BASE_DIR, "dataset_full.memmap")
    META_PATH = os.path.join(BASE_DIR, "dataset_meta.npz")
    
    # 权重保存路径
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_v100")
    
    # --- 硬件配置 ---
    DEVICE = "cuda"
    # V100 节点每个GPU对应 4核CPU
    NUM_WORKERS = 4          
    PIN_MEMORY = True        
    
    # --- 模型参数 ---
    INPUT_CHANNELS = 6
    IMG_SIZE = 224 
    
    # --- 训练参数 ---
    BATCH_SIZE = 64         
    LEARNING_RATE = 3e-4     
    WEIGHT_DECAY = 1e-2      
    EPOCHS = 100             
    
    MOUSE_LOSS_WEIGHT = 20.0 
    CLICK_LOSS_WEIGHT = 1.0
