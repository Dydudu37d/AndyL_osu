import osrparse
import os

# 替换成你刚才下载成功的任意一个文件路径
replay_path = "./osr_dataset/712376_3239866414.osr" 

# 解析回放
try:
    replay = osrparse.Replay.from_path(replay_path)
    
    print("=== 元数据 (Metadata) ===")
    print(f"玩家名称: {replay.username}")
    print(f"连击数 (Combo): {replay.max_combo}")
    print(f"300数: {replay.count_300}")
    print(f"100数: {replay.count_100}")
    print(f"50数:  {replay.count_50}")
    print(f"Miss数: {replay.count_miss}")
    
    print("\n=== 核心回放数据 (Replay Data) ===")
    # 这里的 replay_data 就是解压 LZMA 后得到的坐标列表
    data = replay.replay_data
    
    print(f"总帧数: {len(data)}")
    
    # 打印前 10 帧看看结构
    print("前 10 帧数据 (时间间隔 | X坐标 | Y坐标 | 按键状态):")
    for i, frame in enumerate(data[:10]):
        # 注意: frame.time_delta 是距离上一帧的时间间隔(毫秒)
        # frame.x, frame.y 是 osu! 像素坐标 (0-512, 0-384)
        print(f"Frame {i}: {frame.time_delta}ms | ({frame.x}, {frame.y}) | Keys: {frame.keys}")

except Exception as e:
    print(f"解析失败: {e}")