import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import time
from tensorboardX import SummaryWriter
from osuenv import OsuEnv
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 修改后的ActorCritic类
class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        
        # CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 计算卷积输出尺寸
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out_size = self.cnn(dummy).shape[1]
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        
        # 输出头
        self.actor = nn.Linear(512, n_actions)  # 策略头
        self.critic = nn.Linear(512, 1)        # 价值头

    def forward(self, x):
        # 确保输入格式 [B,C,H,W]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2) if x.size(-1) == 1 else x
        
        # 前向传播
        x = x.float() / 255.0
        features = self.cnn(x)
        latent = self.fc(features)
        
        return torch.softmax(self.actor(latent), self.critic(latent))

class PPOTrainer:
    def __init__(self, env, hyperparams):
        self.env = env
        self.hp = hyperparams
        
        # 初始化模型
        self.model = ActorCritic(
            input_shape=env.observation_space.shape,
            n_actions=env.action_space.n
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hp['lr'])
        self.writer = SummaryWriter()
        
        # 训练状态
        self.episode = 0
        self.best_reward = -float('inf')
        
    def train(self):
        print("Starting training...")
        total_steps = 0
        
        while total_steps < self.hp['max_steps']:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = [], [], [], [], []
            
            # 收集经验
            while len(batch_obs) < self.hp['batch_size']:
                obs = self.env.reset()
                done = False
                ep_rewards = []
                
                for _ in range(self.hp['max_ep_len']):
                    total_steps += 1
                    
                    # 获取动作
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        probs, value = self.model(obs_tensor)
                    dist = Categorical(probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    
                    # 执行动作
                    next_obs, reward, done, _ = self.env.step(action.item())
                    
                    # 存储数据
                    batch_obs.append(obs)
                    batch_acts.append(action)
                    batch_log_probs.append(log_prob)
                    ep_rewards.append(reward)
                    
                    obs = next_obs
                    if done:
                        break
                
                # 计算折扣回报
                batch_rtgs += self._compute_rtgs(ep_rewards)
                batch_lens.append(len(ep_rewards))
                
            # 更新模型
            self._update(
                batch_obs,
                batch_acts,
                batch_log_probs,
                batch_rtgs,
                batch_lens
            )
            
            # 记录日志
            mean_reward = np.mean([sum(r) for r in batch_rtgs])
            self.writer.add_scalar('Reward/mean', mean_reward, self.episode)
            
            # 保存模型
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                torch.save(self.model.state_dict(), f"best_model_{self.episode}.pth")
            
            self.episode += 1
            print(f"Episode {self.episode} | Avg Reward: {mean_reward:.2f}")
            
        self.env.close()
    
    def _compute_rtgs(self, rewards):
        rtgs = []
        discounted_reward = 0
        
        for r in reversed(rewards):
            discounted_reward = r + self.hp['gamma'] * discounted_reward
            rtgs.insert(0, discounted_reward)
        
        return rtgs
    
    def _update(self, batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens):
        # 转换为张量
        obs_tensor = torch.tensor(batch_obs, dtype=torch.float32).to(device)
        acts_tensor = torch.tensor(batch_acts, dtype=torch.float32).to(device)
        log_probs_old = torch.tensor(batch_log_probs, dtype=torch.float32).to(device)
        rtgs_tensor = torch.tensor(batch_rtgs, dtype=torch.float32).to(device)
        
        # 优化次数
        for _ in range(self.hp['update_epochs']):
            # 计算新概率
            probs, values = self.model(obs_tensor)
            dist = Categorical(probs)
            log_probs_new = dist.log_prob(acts_tensor.squeeze())
            
            # 计算比率和优势
            ratios = torch.exp(log_probs_new - log_probs_old)
            advantages = rtgs_tensor - values.squeeze()
            
            # 计算损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.hp['clip'], 1+self.hp['clip']) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = nn.MSELoss()(values.squeeze(), rtgs_tensor)
            
            entropy = dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            # 记录损失
            self.writer.add_scalar('Loss/actor', actor_loss.item(), self.episode)
            self.writer.add_scalar('Loss/critic', critic_loss.item(), self.episode)
            self.writer.add_scalar('Loss/total', loss.item(), self.episode)

if __name__ == "__main__":
    # 初始化环境
    env = OsuEnv(screen_region=(0, 40, 1920, 1040))
    
    # 超参数配置
    hyperparams = {
        'lr': 3e-4,
        'gamma': 0.99,
        'clip': 0.2,
        'update_epochs': 4,
        'batch_size': 4000,
        'max_steps': 1e6,
        'max_ep_len': 3000
    }
    
    # 创建训练器并开始训练
    trainer = PPOTrainer(env, hyperparams)
    trainer.train()
