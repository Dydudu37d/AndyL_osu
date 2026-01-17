import os
import shutil
import osrparse
from osrparse import Replay, Mod

# ================= 配置区域 =================
SOURCE_DIR = "./osr_dataset"       # 你的爬虫下载目录
OUTPUT_DIR = "./dataset_clean"     # 清洗后存放目录
TRASH_DIR = "./dataset_trash"      # 垃圾桶（不符合条件的文件移到这里）

# 筛选阈值
MAX_MISS_COUNT = 10                 # 允许的最大 Miss 数 (IDM 训练建议 0-2)
MIN_COMBO = 100                    # 最小连击数 (太短的不要)
SEPARATE_DT = True                 # 是否把 DT (DoubleTime) 单独分文件夹存放

# 定义“脏” Mods (位掩码)
# Relax, AutoPilot, Auto, Cinema, SpunOut
FORBIDDEN_MODS = Mod.Relax | Mod.Autopilot | Mod.Autoplay | Mod.Cinema | Mod.SpunOut
# ===========================================

def setup_dirs():
    """创建必要的目录结构"""
    for d in [OUTPUT_DIR, TRASH_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    # 如果需要分离 DT
    if SEPARATE_DT:
        dt_dir = os.path.join(OUTPUT_DIR, "DT")
        nm_dir = os.path.join(OUTPUT_DIR, "NM_HD") # NoMod / Hidden
        if not os.path.exists(dt_dir): os.makedirs(dt_dir)
        if not os.path.exists(nm_dir): os.makedirs(nm_dir)

def get_target_subfolder(replay):
    """根据 Mod 决定存放子目录"""
    if not SEPARATE_DT:
        return ""
    
    # 检查是否包含 DT 或 NC (Nightcore 也是变加速)
    if (replay.mods & Mod.DoubleTime) or (replay.mods & Mod.Nightcore):
        return "DT"
    return "NM_HD"

def inspect_replay(file_path):
    """
    检查单个回放文件 (纯整数防爆版)
    返回: (是否通过, 拒绝原因, 解析后的对象)
    """
    try:
        # 1. 尝试解析
        r = Replay.from_path(file_path)
    except Exception as e:
        return False, f"Corrupted ({str(e)})", None

    # === 强制修复 1: 游戏模式 (GameMode) ===
    # 无论库里叫 Standard, OSU 还是 STD，标准模式的 ID 永远是 0
    # 我们先尝试把 r.mode 转成 int (防止它是 Enum 对象)
    try:
        mode_int = int(r.mode)
    except:
        # 如果转不了 int，说明它是某种奇怪的对象，尝试取 .value
        mode_int = r.mode.value if hasattr(r.mode, 'value') else -1
        
    if mode_int != 0: 
        return False, f"Wrong Mode (ID: {mode_int})", r

    # === 强制修复 2: Mods (模组) ===
    # 同样强制转为 int，避开 Mod.Relax 这种属性可能不存在的问题
    try:
        mods_value = int(r.mods)
    except:
        mods_value = r.mods.value if hasattr(r.mods, 'value') else 0

    # 定义“脏” Mods 的位掩码 (Hardcode 是最稳的)
    # Relax(128) + AutoPilot(8192) + Auto(2048) + Cinema(4194304) + SpunOut(4096)
    FORBIDDEN_MASK = 128 | 8192 | 2048 | 4194304 | 4096
    
    if mods_value & FORBIDDEN_MASK:
        return False, f"Forbidden Mod (Mask: {mods_value & FORBIDDEN_MASK})", r

    # 4. 检查表现质量
    if r.count_miss > MAX_MISS_COUNT:
        return False, f"Too many misses ({r.count_miss})", r

    # 5. 检查长度
    if r.max_combo < MIN_COMBO:
        return False, f"Combo too low ({r.max_combo})", r

    return True, "OK", r

def main():
    setup_dirs()
    
    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".osr")]
    total = len(files)
    print(f"=== 开始清洗数据: 共 {total} 个文件 ===")
    
    stats = {
        "ok": 0,
        "trash": 0,
        "reasons": {}
    }

    for idx, filename in enumerate(files):
        src_path = os.path.join(SOURCE_DIR, filename)
        
        # 执行检查
        passed, reason, replay_obj = inspect_replay(src_path)
        
        if passed:
            stats["ok"] += 1
            # 决定目标文件夹
            subfolder = get_target_subfolder(replay_obj)
            dst_dir = os.path.join(OUTPUT_DIR, subfolder) if SEPARATE_DT else OUTPUT_DIR
            dst_path = os.path.join(dst_dir, filename)
            
            # 复制/移动文件
            shutil.copy2(src_path, dst_path)
            # print(f"[+] 保留: {filename}") # 刷屏可以注释掉
            
        else:
            stats["trash"] += 1
            dst_path = os.path.join(TRASH_DIR, filename)
            shutil.move(src_path, dst_path)
            
            # 记录拒绝原因统计
            base_reason = reason.split('(')[0].strip()
            stats["reasons"][base_reason] = stats["reasons"].get(base_reason, 0) + 1
            print(f"[-] 剔除: {filename} -> {reason}")

        # 进度条
        if (idx + 1) % 100 == 0:
            print(f"进度: {idx + 1}/{total} ...")

    print("\n" + "="*30)
    print(f"清洗完成！")
    print(f"保留: {stats['ok']}")
    print(f"剔除: {stats['trash']}")
    print("剔除原因详情:")
    for r, c in stats["reasons"].items():
        print(f"  - {r}: {c}")
    print(f"干净数据已存入: {OUTPUT_DIR}")
    print("="*30)

if __name__ == "__main__":
    main()