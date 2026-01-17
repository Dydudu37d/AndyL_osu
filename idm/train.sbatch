#!/bin/bash
#SBATCH --job-name=osu_ddp
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=4                    # DDP 模式: 4进程对应4卡
#SBATCH --gpus-per-node=4
#SBATCH --qos=rush-4gpu
#SBATCH --output=logs/%j_train.out
#SBATCH --error=logs/%j_train.err

# --- 1. 环境准备 ---
source /opt/devtools/anaconda3/bin/activate train_osu_idm

# 注意：这里删除了 mps_mapping 相关的 source，以避免干扰 torchrun

# --- 2. 数据搬运 (核心修复部分) ---
# 定义本地缓存的完整路径 (包含子文件夹)
LOCAL_CACHE="/cache_local/testgroup01/wangzirui/output_dataset"
# 定义全局存储的真实路径 (请务必修改为您存放 dataset_meta.npz 的实际路径)
GLOBAL_DATA="/home/testgroup01/wangzirui/" 

# 如果本地没有数据，则从全局存储拷贝
if [ ! -d "$LOCAL_CACHE" ]; then
    echo "正在搬运数据到: $LOCAL_CACHE"
    mkdir -p "$LOCAL_CACHE"
    cp "$GLOBAL_DATA/dataset_full.memmap" "$LOCAL_CACHE/"
    cp "$GLOBAL_DATA/dataset_meta.npz" "$LOCAL_CACHE/"
fi

# 【关键修复】设置环境变量，告诉 config.py 数据到底在哪
# config.py 中的 os.getenv("LOCAL_DATA_DIR") 会读取这个值
export LOCAL_DATA_DIR="$LOCAL_CACHE"

# --- 3. 运行训练 ---
echo "启动监控..."
nvidia-smi dmon -s pucvmte -o T > logs/${SLURM_JOB_ID}_gpu_monitor.log &

echo "开始训练..."
# 启动 DDP
torchrun --nproc_per_node=4 train.py

exit