import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import pygetwindow as gw
import struct
import time
import threading
import asyncio
import websockets
from collections import deque

class OsuEnv(gym.Env):
    """
    osu!lazer 强化学习环境
    基于 WebSocket Mod 进行控制和数据读取，基于 mss 进行视觉捕获。
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, host="127.0.0.1", port=8765, obs_shape=(84, 84), render_mode="human"):
        super().__init__()
        
        self.host = host
        self.port = port
        self.obs_shape = obs_shape
        self.render_mode = render_mode
        self.window_title = "osu!"

        # === 1. 定义动作空间 ===
        # [x, y, click]
        # x, y: [-1, 1] 对应屏幕坐标
        # click: > 0 为按下，<= 0 为松开
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # === 2. 定义观察空间 ===
        # 屏幕截图: (H, W, 1) 灰度图 或 (H, W, 3) 彩色图
        # 这里使用灰度图以加速训练
        self.observation_space = spaces.Box(low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8)

        # === 3. 初始化状态变量 ===
        self.monitor = self._find_game_window()
        self.sct = mss.mss()
        self.last_score = 0
        self.last_combo = 0
        self.last_acc = 0.0
        
        # 线程同步相关
        self._latest_telemetry = {"score": 0, "combo": 0, "acc": 0.0}
        self._action_queue = deque(maxlen=1)
        self._stop_event = threading.Event()
        self._telemetry_lock = threading.Lock()
        self._ws_connected = False

        # === 4. 启动 WebSocket 后台线程 ===
        self.ws_thread = threading.Thread(target=self._ws_loop_entry, daemon=True)
        self.ws_thread.start()
        
        # 等待连接建立
        print("等待 WebSocket 连接...")
        while not self._ws_connected:
            time.sleep(0.1)
        print("环境初始化完成！")

    def _find_game_window(self):
        try:
            win = gw.getWindowsWithTitle(self.window_title)[0]
            return {"top": win.top, "left": win.left, "width": win.width, "height": win.height}
        except IndexError:
            print(f"未找到 '{self.window_title}' 窗口，使用全屏模式。")
            return {"top": 0, "left": 0, "width": 1920, "height": 1080}

    # --- WebSocket 异步逻辑封装 ---
    def _ws_loop_entry(self):
        asyncio.run(self._async_ws_loop())

    async def _async_ws_loop(self):
        uri = f"ws://{self.host}:{self.port}"
        async with websockets.serve(self._ws_handler, self.host, self.port):
            print(f"WebSocket 服务器监听中: {uri}")
            self._ws_connected = True
            await self._stop_event.wait() # 保持运行直到外部停止

    async def _ws_handler(self, websocket):
        # 启动接收任务
        recv_task = asyncio.create_task(self._recv_telemetry(websocket))
        
        try:
            while not self._stop_event.is_set():
                # 发送动作
                if self._action_queue:
                    dx, dy, pressed = self._action_queue.pop()
                    # 直接发送原始动作数据给 Mod
                    # Mod 内部会乘以 SENSITIVITY (灵敏度系数)
                    data = struct.pack('<IIff?', 13, 0, float(dx), float(dy), pressed > 0)
                    await websocket.send(data)
                
                # 60Hz 刷新率控制
                await asyncio.sleep(0.016)
        except Exception as e:
            print(f"WS Error: {e}")
        finally:
            recv_task.cancel()

    async def _recv_telemetry(self, websocket):
        try:
            async for message in websocket:
                # 旧协议: Length(I), Score(q), Acc(d), Combo(i) = 24 bytes
                # 新协议: Length(I), Score(q), Acc(d), Combo(i), IsDead(?) = 25 bytes
                # 注意：message 包含头部 Length(4 bytes)，所以实际 buffer 是 4 + 21 = 25 bytes
                
                if len(message) >= 25:
                    # 跳过头部的 4 字节 Length
                    # 解包: Score(q), Acc(d), Combo(i), IsDead(?)
                    score, acc, combo, is_dead = struct.unpack('<qdi?', message[4:25])
                    
                    with self._telemetry_lock:
                        self._latest_telemetry = {
                            "score": score,
                            "acc": acc,
                            "combo": combo,
                            "is_dead": is_dead  # 新增
                        }
        except Exception as e:
            print(f"Telemetry Error: {e}")

    # --- Gym 接口实现 ---
    
    def step(self, action):
        # 1. 发送动作到队列
        self._action_queue.append(action)
        
        # 2. 等待一帧 (简单的同步，实际训练可能不需要 sleep，取决于训练速度)
        # time.sleep(0.01) 

        # 3. 获取观察 (Vision)
        raw_screen = np.array(self.sct.grab(self.monitor))
        # 转换为灰度图并缩放
        gray = cv2.cvtColor(raw_screen, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(gray, (self.obs_shape[0], self.obs_shape[1]))
        # 增加通道维度 (H, W, 1)
        observation = np.expand_dims(resized, axis=-1)

        # 4. 获取状态 (Telemetry) & 计算奖励
        with self._telemetry_lock:
            current_score = self._latest_telemetry["score"]
            current_combo = self._latest_telemetry["combo"]
            current_acc = self._latest_telemetry["acc"]
            is_dead = self._latest_telemetry.get("is_dead", False) # 获取死亡状态

        # --- 奖励函数设计 ---
        reward = 0.0
        
        # A. 得分奖励 (缩放一下，防止数字太大)
        score_diff = current_score - self.last_score
        if score_diff > 0:
            reward += score_diff * 0.01 

        # B. 断连惩罚 (Combo 突然归零且之前不为0)
        if self.last_combo > 5 and current_combo == 0:
            reward -= 10.0 # 大惩罚
        
        # C. Combo 增长奖励 (鼓励连击)
        if current_combo > self.last_combo:
            reward += 0.1 * current_combo # 连击越高奖励越多

        # 更新历史状态
        self.last_score = current_score
        self.last_combo = current_combo
        self.last_acc = current_acc

        # 5. 判断结束 (Terminated)
        # 难点：Mod 目前没有发送 "Failed" 或 "Finished" 信号
        # 临时方案：如果 10秒内 分数没变且 Combo 为 0，认为结束？
        # 或者训练时手动重开。这里暂时设为 False。
        terminated = False 
        truncated = False
        # 5. 判断结束 (Terminated)
        terminated = False
        
        # 如果 Mod 告诉我们要死了，那就结束回合
        if is_dead:
            terminated = True
            reward -= 50.0 # 死亡给予巨大惩罚
        
        info = {
            "score": current_score,
            "combo": current_combo,
            "acc": current_acc
        }

        if self.render_mode == "human":
            cv2.imshow("Osu! RL Agent View", resized)
            cv2.waitKey(1)

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 重置内部状态
        self.last_score = 0
        self.last_combo = 0
        self.last_acc = 0.0
        
        # 注意：这里无法自动重启游戏，需要外部配合（如 Auto-Retry Mod 或 手动重启）
        # 可以在这里添加一个等待，直到检测到分数归零（表示新的一局开始）
        print("Resetting... 等待新游戏开始 (请手动重开或使用 AutoRetry)...")
        
        # 简单的逻辑：获取一张初始截图
        raw_screen = np.array(self.sct.grab(self.monitor))
        gray = cv2.cvtColor(raw_screen, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(gray, (self.obs_shape[0], self.obs_shape[1]))
        observation = np.expand_dims(resized, axis=-1)
        
        return observation, {}

    def close(self):
        self._stop_event.set()
        self.ws_thread.join()
        cv2.destroyAllWindows()