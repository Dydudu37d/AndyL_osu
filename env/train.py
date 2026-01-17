import os
import time
from osu_env import OsuEnv  # 导入我们定义的 osu! 环境
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def train():
    # 1. 创建环境
    # obs_shape=(84, 84) 是 RL 处理图像的标准尺寸
    env = OsuEnv(obs_shape=(84, 84), render_mode="rgb_array")
    
    # 2. 包装环境 (向量化)
    # VecFrameStack 非常重要：它把连续的 4 帧画面叠在一起传给 AI
    # 这样 AI 才能通过画面感受到物体的“移动速度”和“方向”
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

    # 3. 设置模型保存路径
    model_dir = "models/ppo_osu_relative"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 4. 定义 PPO 模型
    # CnnPolicy: 专门处理图像输入的策略
    # verbose=1: 打印训练进度
    # device="auto": 自动检测 GPU (cuda)
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=3e-4,     # 学习率
        n_steps=2048,           # 每次更新前收集的步数
        batch_size=64,          # 更新时的批大小
        n_epochs=10,            # 每次更新的迭代次数
        gamma=0.99,             # 折扣因子（对未来奖励的重视程度）
        verbose=1,
        tensorboard_log=log_dir,
        device="auto"
    )

    # 5. 设置自动保存回调（每 10,000 步保存一次模型）
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="osu_model"
    )

    print("--- 训练即将开始 ---")
    print("请确保：")
    print("1. osu! 已经启动并开启了 WebSocket Control Mod")
    print("2. 已经进入游戏画面（最好开启 NoFail Mod）")
    print("3. 按 Ctrl+C 可以随时中断并保存模型")
    
    try:
        # 6. 开始训练 (total_timesteps 代表总步数，越多学得越久)
        model.learn(
            total_timesteps=500000, 
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n检测到中断，正在保存当前模型...")
    finally:
        # 7. 保存最终模型
        model.save(f"{model_dir}/osu_model_final")
        print(f"模型已保存至 {model_dir}")
        env.close()

if __name__ == "__main__":
    train()