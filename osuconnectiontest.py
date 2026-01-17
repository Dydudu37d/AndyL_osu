import asyncio
import websockets
import struct
import math
import time
import numpy as np
import cv2
import mss
import pygetwindow as gw

# ================= 配置区域 =================
WINDOW_TITLE = "osu!" # osu!lazer 的窗口标题通常是 "osu!"
# ===========================================

async def handler(websocket):
    print(f"Mod 已连接: {websocket.remote_address}")

    # --- 1. 查找游戏窗口位置 ---
    try:
        # 尝试找到 osu! 窗口
        win = gw.getWindowsWithTitle(WINDOW_TITLE)[0]
        # 定义截屏区域 (left, top, width, height)
        monitor = {"top": win.top, "left": win.left, "width": win.width, "height": win.height}
        print(f"已锁定游戏窗口: {monitor}")
    except IndexError:
        print(f"错误: 未找到名为 '{WINDOW_TITLE}' 的窗口。")
        print("将使用全屏截图作为备选方案。")
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    except Exception as e:
        print(f"窗口检测出错: {e}")
        return

    # --- 2. 接收遥测数据 (Score/Combo) 的协程 ---
    async def receive_telemetry():
        try:
            async for message in websocket:
                # 结构: Length(I), Score(q), Accuracy(d), Combo(i)
                # 总长度 24 bytes (4+8+8+4)
                if len(message) >= 24:
                    _, score, acc, combo = struct.unpack('<Iqdi', message[:24])
                    # 在控制台动态打印，不换行
                    print(f"\r[状态] 得分:{score:<10} 连击:{combo:<5} 准度:{acc:.2%}", end="")
        except Exception as e:
            pass

    # 启动接收任务
    asyncio.create_task(receive_telemetry())

    # --- 3. 主循环：截屏 + 决策 + 控制 ---
    start_time = time.time()
    
    # 初始化截屏工具
    with mss.mss() as sct:
        try:
            while True:
                # === A. 获取画面 (Vision) ===
                # grab() 返回的是原始像素数据
                screenshot = sct.grab(monitor)
                
                # 将数据转换为 numpy 数组 (BGRA 格式)
                img = np.array(screenshot)
                
                # 转为 OpenCV 格式 (去掉 Alpha 通道，保留 BGR)
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # [可选] 缩小图片以加速 AI 处理 (例如缩小到 256x192)
                # ai_input = cv2.resize(frame, (256, 192))

                # [演示] 显示当前的画面，证明我们获取到了
                # 实际跑 AI 训练时请注释掉下面这两行，因为显示窗口会拖慢速度
                cv2.imshow("AI Vision Input", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # === B. AI 决策 (Logic) ===
                # 这里依然用画圆作为测试
                t = time.time() - start_time
                
                # 让光标绕屏幕中心画圆
                # 注意：osu! 逻辑分辨率是 512x384
                cursor_x = 256 + 100 * math.cos(t * 3)
                cursor_y = 192 + 100 * math.sin(t * 3)
                
                # 每秒点击 4 次
                is_pressed = (int(t * 4) % 2) == 0

                # === C. 发送指令 (Action) ===
                # 结构: Length(13), ID(0), X, Y, Pressed
                data = struct.pack('<IIff?', 13, 0, float(cursor_x), float(cursor_y), is_pressed)
                await websocket.send(data)
                
                # 控制循环频率，例如 60Hz
                # 注意：截屏操作本身需要几毫秒，这里的 sleep 可以适当减小
                await asyncio.sleep(0.01)

        except websockets.exceptions.ConnectionClosed:
            print("\n连接已断开")
        except Exception as e:
            print(f"\n发生错误: {e}")
        finally:
            cv2.destroyAllWindows()

async def main():
    print("正在启动 Python AI 服务器...")
    print(f"请确保游戏已打开，且 Mod 已启用。")
    async with websockets.serve(handler, "127.0.0.1", 8765):
        print("服务器监听中: ws://127.0.0.1:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())