import os
import glob
import numpy as np
from tqdm import tqdm

# --- 配置 ---
INPUT_DIR = "./output_dataset_500k"  # 你原本存放 npz 的地方
OUTPUT_FILE = "./dataset_full.memmap" # 生成的超级大文件路径
META_FILE = "./dataset_meta.npz"      # 存放元数据(索引/动作)

def convert():
    print(f"🔍 Scanning {INPUT_DIR}...")
    npz_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npz")))
    if not npz_files:
        print("❌ No .npz files found!")
        return

    # 1. 先扫一遍，计算总样本数 (Total Count)
    total_samples = 0
    sample_shape = None
    
    print("📊 Calculating total dataset size...")
    # 为了速度，我们只读第一个文件获取形状，其他的只读 header
    # 注意：这里假设所有图片的 shape 是一样的 (128x128 或 224x224)
    first_data = np.load(npz_files[0])
    sample_shape = first_data['sequences'].shape[1:] # (6, H, W)
    dtype = first_data['sequences'].dtype
    
    # 快速统计总数
    for f in tqdm(npz_files):
        try:
            # 只读 header 信息，不加载数据，速度极快
            with np.load(f) as data:
                total_samples += data['actions'].shape[0]
        except:
            pass
            
    print(f"\n✅ Total Samples: {total_samples}")
    print(f"✅ Data Shape: {sample_shape}")
    print(f"✅ Estimated Size: {total_samples * np.prod(sample_shape) / 1024**3:.2f} GB")

    # 2. 创建一个内存映射文件 (Memmap)
    # 这会在硬盘上预分配一个巨大的文件
    fp = np.memmap(OUTPUT_FILE, dtype=dtype, mode='w+', shape=(total_samples, *sample_shape))
    
    # 我们把所有的 actions (标签) 读到内存里存成一个单独的小文件，因为标签很小
    all_actions = []
    
    # 3. 开始搬运数据
    print("🚀 Converting data to raw memmap (Sequential Write)...")
    current_idx = 0
    
    for f in tqdm(npz_files):
        try:
            with np.load(f) as data:
                seqs = data['sequences']
                acts = data['actions']
                
                n_batch = len(seqs)
                
                # 直接写入硬盘映射区
                fp[current_idx : current_idx + n_batch] = seqs
                all_actions.append(acts)
                
                current_idx += n_batch
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    # 刷入硬盘
    fp.flush()
    del fp # 关闭句柄
    
    # 保存标签数据
    print("💾 Saving metadata/actions...")
    all_actions = np.concatenate(all_actions, axis=0)
    np.savez(META_FILE, actions=all_actions, shape=sample_shape, total=total_samples)
    
    print("\n🎉 Conversion Complete!")
    print(f"Path: {OUTPUT_FILE}")

if __name__ == "__main__":
    convert()