import numpy as np
from PIL import Image
import mss
import gym
from gym import spaces
import mouse
import websockets
import asyncio
import json
import threading

class OsuEnv(gym.Env):
    """Custom osu! environment for reinforcement learning"""
    
    def __init__(self, ws_url="ws://127.0.0.1:24050/websocket/v2", 
                 screen_region=(0, 0, 1920, 1080), delta=10):
        super(OsuEnv, self).__init__()
        
        # 动作空间：上、下、左、右、点击（5个离散动作）
        self.action_space = spaces.Discrete(5)
        
        # 状态空间：84x84的灰度图像
        self.observation_space = spaces.Box(low=0, high=255, 
                                          shape=(84, 84, 1), dtype=np.uint8)
        
        # 屏幕捕捉设置
        self.sct = mss.mss()
        self.screen_region = screen_region
        self.delta = delta  # 鼠标移动步长
        
        # WebSocket连接
        self.ws_url = ws_url
        self.ws = None
        self.game_state = {}
        self.running = False
        
        # 奖励相关参数
        self.last_hits = {'300': 0, '100': 0, '50': 0, '0': 0}
        self.last_health = 0
        self.combo = 0

    def _process_image(self, img):
        """预处理屏幕截图"""
        img = Image.frombytes('RGB', img.size, img.rgb)
        img = img.convert('L').resize((84, 84))  # 转换为灰度并缩小
        return np.array(img).reshape(84, 84, 1)

    def _get_reward(self, current_hits):
        """计算奖励函数"""
        reward = 0
        
        # 基于命中判定的奖励
        hit_rewards = {
            '300': 1.0,
            '100': 0.5,
            '50': 0.2,
            '0': -1.0
        }
        
        for key in hit_rewards:
            delta = current_hits[key] - self.last_hits[key]
            reward += delta * hit_rewards[key]
        
        # 连击奖励
        if self.game_state.get('combo', 0) > self.combo:
            reward += 0.1
        self.combo = self.game_state.get('combo', 0)
        
        # 健康值变化奖励
        current_health = self.game_state.get('healthBar', {}).get('normal', 0)
        reward += (current_health - self.last_health) * 2
        self.last_health = current_health
        
        # 滑条中断惩罚
        if self.game_state.get('hits', {}).get('sliderBreaks', 0) > 0:
            reward -= 2.0
        
        return reward

    async def _ws_handler(self):
        """WebSocket数据接收"""
        async with websockets.connect(self.ws_url) as websocket:
            while self.running:
                try:
                    data = await websocket.recv()
                    self.game_state = json.loads(data)
                except:
                    break

    def reset(self):
        """重置环境"""
        # 初始化WebSocket连接
        self.running = True
        self.ws_thread = threading.Thread(target=lambda: asyncio.run(self._ws_handler()))
        self.ws_thread.start()
        
        # 重置游戏状态
        self.last_hits = {'300': 0, '100': 0, '50': 0, '0': 0}
        self.last_health = 0
        self.combo = 0
        
        # 获取初始状态
        return self._get_obs()

    def _get_obs(self):
        """获取当前观察"""
        screen = self.sct.grab(self.screen_region)
        return self._process_image(screen)

    def step(self, action):
        """执行动作"""
        # 执行鼠标操作
        if action == 0:    # 上
            mouse.move(0, -self.delta, absolute=False)
        elif action == 1:  # 下
            mouse.move(0, self.delta, absolute=False)
        elif action == 2:  # 左
            mouse.move(-self.delta, 0, absolute=False)
        elif action == 3:  # 右
            mouse.move(self.delta, 0, absolute=False)
        elif action == 4:  # 点击
            mouse.click()
        
        # 获取新状态
        obs = self._get_obs()
        
        # 计算奖励
        current_hits = self.game_state.get('hits', {})
        reward = self._get_reward(current_hits)
        self.last_hits = current_hits.copy()
        
        # 检查终止条件
        done = self.game_state.get('healthBar', {}).get('normal', 0) <= 0
        
        return obs, reward, done, {}

    def close(self):
        """清理环境"""
        self.running = False
        if self.ws_thread.is_alive():
            self.ws_thread.join()
