#!/bin/bash
#SBATCH --job-name=osu_ddp_home
#SBATCH --partition=4V100PX           # 依然建议用 PX 分区，排队可能快些，或者改回 4V100 [cite: 161, 163]
#SBATCH --nodes=1
#SBATCH --ntasks=4                    # DDP 必须设为 4 [cite: 197]
#SBATCH --gpus-per-node=4
#SBATCH --qos=rush-4gpu               # 4卡高优先级 [cite: 46]
#SBATCH --output=logs/%j_home.out
#SBATCH --error=logs/%j_home.err

# --- 1. 环境准备 ---
source /opt/devtools/anaconda3/bin/activate train_osu_idm

# --- 2. 路径配置 (直接指向 Home) ---
# 【请修改此处】为您存放数据的真实目录
# 注意：数据必须已经在 /home 下，不能在登录节点的 /cache_local 里
DATA_PATH="/home/testgroup01/wangzirui/"

# 模型保存路径
CKPT_PATH="/home/testgroup01/wangzirui/checkpoints_v100"

# --- 3. 安全检查 ---
if [ ! -f "$DATA_PATH/dataset_meta.npz" ]; then
    echo "❌ 错误：在路径 $DATA_PATH 下没找到数据集！"
    echo "⚠️ 提醒：计算节点无法访问登录节点的 /cache_local，请确保数据真的在 /home 目录下。"
    exit 1
fi

# --- 4. 设置环境变量 ---
# 直接告诉 Python 脚本去 Home 目录读数据
export LOCAL_DATA_DIR="$DATA_PATH"
export CHECKPOINT_DIR="$CKPT_PATH"

# 确保权重目录存在
mkdir -p "$CKPT_PATH"

# --- 5. 启动训练 ---
echo "✅ 模式：直连 Global Home 存储，跳过本地缓存搬运"
echo "📂 数据路径: $DATA_PATH"

# 启动监控
nvidia-smi dmon -s pucvmte -o T > logs/${SLURM_JOB_ID}_gpu_monitor.log &

# 启动 DDP 训练
torchrun --nproc_per_node=4 train.py

exit
