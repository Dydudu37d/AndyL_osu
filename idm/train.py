import os
import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

from config import Config
from model import OsuIDM

def setup_ddp():
    # SAI 配合 Slurm 会自动设置 MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE 等环境变量
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

class FastOsuDataset(Dataset):
    def __init__(self, memmap_path, meta_path):
        meta = np.load(meta_path)
        self.total_samples = int(meta['total'])
        self.shape = tuple(meta['shape'])
        self.actions = torch.from_numpy(meta['actions'].astype(np.float32))
        # 使用 mode='r' 防止意外修改数据
        self.data_memmap = np.memmap(memmap_path, dtype=np.uint8, mode='r', shape=(self.total_samples, *self.shape))

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 必须 copy，否则转 Tensor 可能报错或产生负 stride 问题
        seq_data = np.array(self.data_memmap[idx], copy=True)
        seq_tensor = torch.from_numpy(seq_data).float() / 255.0
        label_tensor = self.actions[idx]
        return seq_tensor, label_tensor

def find_latest_checkpoint(checkpoint_dir):
    """查找目录中最新的检查点文件"""
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    # 假设文件名格式为 "osu_ddp_epoch_{epoch}.pth"
    files = glob.glob(os.path.join(checkpoint_dir, "osu_ddp_epoch_*.pth"))
    if not files:
        return None, 0
    
    # 提取 epoch 数字并排序
    latest_file = max(files, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))
    latest_epoch = int(re.search(r'epoch_(\d+)', latest_file).group(1))
    return latest_file, latest_epoch

def validate(model, val_loader, criterion_mouse, criterion_click, local_rank):
    """验证循环"""
    model.eval()
    total_loss = torch.tensor(0.0).to(local_rank)
    count = torch.tensor(0.0).to(local_rank)
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(local_rank, non_blocking=True)
            labels = labels.to(local_rank, non_blocking=True)
            
            # 验证集通常不需要 autocast，除非显存非常紧张，但保持一致较好
            with autocast():
                pred_mouse, pred_click = model(imgs)
                loss = (criterion_mouse(pred_mouse, labels[:, :2]) * Config.MOUSE_LOSS_WEIGHT) + \
                       (criterion_click(pred_click, labels[:, 2:3]) * Config.CLICK_LOSS_WEIGHT)
            
            total_loss += loss
            count += 1
    
    # 汇总所有进程的 Loss
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    
    avg_loss = total_loss / count
    return avg_loss.item()

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, path):
    """保存包含所有状态的检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict()
    }
    torch.save(checkpoint, path)

def train():
    local_rank = setup_ddp()
    is_main_process = (dist.get_rank() == 0)
    
    if is_main_process:
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        print(f"🔥 Distributed training started on {dist.get_world_size()} GPUs")

    # 1. 加载数据集
    dataset = FastOsuDataset(Config.MEMMAP_PATH, Config.META_PATH)
    
    # 2. 数据切分 (FIX: 使用固定种子确保所有 rank 切分一致)
    train_size = int(0.98 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42) # 固定种子
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    
    # 训练集 Sampler
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    # 验证集 Sampler (shuffle=False, 确保验证结果稳定)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=Config.BATCH_SIZE, 
        sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        sampler=val_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    # 3. 初始化模型
    model = OsuIDM().to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    # OneCycleLR 需要完整的 steps 数量
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=Config.LEARNING_RATE, 
        steps_per_epoch=len(train_loader), 
        epochs=Config.EPOCHS
    )
    
    scaler = GradScaler()
    criterion_mouse = nn.MSELoss()
    criterion_click = nn.BCEWithLogitsLoss()

    # 4. 尝试加载检查点
    start_epoch = 0
    latest_checkpoint_path, latest_epoch = find_latest_checkpoint(Config.CHECKPOINT_DIR)
    
    if latest_checkpoint_path:
        if is_main_process:
            print(f"🔄 Resuming from checkpoint: {latest_checkpoint_path} (Epoch {latest_epoch})")
        
        # map_location 必须指定为当前 GPU，防止爆显存
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(latest_checkpoint_path, map_location=map_location)
        
        # 兼容旧版本（如果之前只保存了 model state dict）
        if 'model_state_dict' in checkpoint:
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] # 从保存的 epoch 开始，意味着该 epoch 已完成，接下来跑 epoch+1
            
            # 如果 OneCycleLR 是按 step 走的，我们可能需要手动 step scheduler 到正确位置
            # 但 pytorch 的 load_state_dict 通常会处理好 last_epoch
        else:
            # 兼容以前只保存了 state_dict 的情况
            if is_main_process:
                print("⚠️ Found legacy checkpoint (weights only). Optimizer state will be reset.")
            model.module.load_state_dict(checkpoint)
            start_epoch = latest_epoch # 假设文件名里的 epoch 是已经跑完的

    # 5. 训练循环
    # 如果 start_epoch = 10，range(10, 100) 会从 Epoch 11 开始跑（打印显示为 Epoch 11）
    for epoch in range(start_epoch, Config.EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        
        loader = tqdm(train_loader, desc=f"Train Epoch {epoch+1}") if is_main_process else train_loader
        
        for imgs, labels in loader:
            imgs, labels = imgs.to(local_rank, non_blocking=True), labels.to(local_rank, non_blocking=True)
            
            with autocast():
                pred_mouse, pred_click = model(imgs)
                loss = (criterion_mouse(pred_mouse, labels[:, :2]) * Config.MOUSE_LOSS_WEIGHT) + \
                       (criterion_click(pred_click, labels[:, 2:3]) * Config.CLICK_LOSS_WEIGHT)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # 6. 验证循环 (每个 epoch 结束)
        if is_main_process:
            print(f"🔍 Validating Epoch {epoch+1}...")
        
        val_loss = validate(model, val_loader, criterion_mouse, criterion_click, local_rank)
        
        if is_main_process:
            print(f"📉 Epoch {epoch+1} | Val Loss: {val_loss:.6f}")

            # 7. 保存检查点 (使用新格式)
            save_path = os.path.join(Config.CHECKPOINT_DIR, f"osu_ddp_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch+1, save_path)
            
            # 可选：清理旧的 checkpoint 以节省空间
            # prev_path = os.path.join(Config.CHECKPOINT_DIR, f"osu_ddp_epoch_{epoch}.pth")
            # if os.path.exists(prev_path): os.remove(prev_path)

    dist.destroy_process_group()

if __name__ == "__main__":
    train()