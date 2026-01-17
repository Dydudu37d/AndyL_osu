import os
import glob
import cv2
import numpy as np
from osrparse import Replay
from collections import deque
import bisect
from concurrent.futures import ProcessPoolExecutor # 引入多进程池
from tqdm import tqdm # 建议安装 tqdm 显示进度条: pip install tqdm

# --- 配置区域 ---
DATA_DIR = "./ordr_videos_batch"
OUTPUT_DIR = "/cache_local/testgroup01/wangzirui/output_dataset"
IMG_SIZE = (128, 128)
TRIM_END_SECONDS = 5.0
CHUNK_SIZE = 1000
HISTORY_LEN = 5
FUTURE_LEN = 1
TOTAL_SEQ_LEN = HISTORY_LEN + FUTURE_LEN

# 多进程配置
MAX_WORKERS = 16  # 根据你的 CPU 核心数调整 (例如 12400F 可以设 6-12)
PROCESS_FILE_LIMIT = 150 # 限制处理多少个视频 (约等于 50万样本)

def parse_osr_events(osr_path):
    # ... (保持不变) ...
    try:
        replay = Replay.from_path(osr_path)
        events = []
        current_time = 0
        for event in replay.replay_data:
            current_time += event.time_delta
            is_pressed = 1.0 if event.keys > 0 else 0.0
            events.append([current_time, event.x, event.y, is_pressed])
        return np.array(events)
    except Exception:
        return None

def save_chunk(chunk_data, file_basename, chunk_id):
    # ... (保持不变) ...
    if not chunk_data['sequences']: return
    save_path = os.path.join(OUTPUT_DIR, f"{file_basename}_part{chunk_id}.npz")
    seq_arr = np.array(chunk_data['sequences'], dtype=np.uint8)
    act_arr = np.array(chunk_data['actions'], dtype=np.float32)
    np.savez_compressed(save_path, sequences=seq_arr, actions=act_arr)

def process_single_video(args):
    """
    为了适配多进程，将参数打包
    args: (mp4_path, osr_path)
    """
    mp4_path, osr_path = args
    file_basename = os.path.basename(mp4_path).replace('.mp4', '')
    
    osr_data = parse_osr_events(osr_path)
    if osr_data is None: return 0
    osr_times = osr_data[:, 0]

    cap = cv2.VideoCapture(mp4_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    valid_duration_frames = total_frames - int(TRIM_END_SECONDS * fps)
    
    frame_buffer = deque(maxlen=TOTAL_SEQ_LEN)
    chunk_buffer = {'sequences': [], 'actions': []}
    chunk_counter = 0
    frame_idx = 0
    local_collected = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= valid_duration_frames:
            break
            
        current_ms = (frame_idx / fps) * 1000.0
        
        # 预处理
        frame_resized = cv2.resize(frame, IMG_SIZE)
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        frame_buffer.append((frame_gray, current_ms))
        
        if len(frame_buffer) == TOTAL_SEQ_LEN:
            state_time = frame_buffer[HISTORY_LEN - 1][1]
            next_time = frame_buffer[HISTORY_LEN][1]
            
            idx_now = bisect.bisect_left(osr_times, state_time)
            idx_next = bisect.bisect_left(osr_times, next_time)
            
            if idx_now < len(osr_data) and idx_next < len(osr_data):
                start_rec = osr_data[idx_now]
                end_rec = osr_data[idx_next]
                
                dx = (end_rec[1] - start_rec[1]) / 512.0
                dy = (end_rec[2] - start_rec[2]) / 384.0
                
                press_interval = osr_data[idx_now:idx_next+1, 3]
                is_pressed = 1.0 if np.any(press_interval > 0.5) else 0.0
                
                img_stack = np.array([item[0] for item in frame_buffer])
                chunk_buffer['sequences'].append(img_stack)
                chunk_buffer['actions'].append([dx, dy, is_pressed])
                
                local_collected += 1
                
                if len(chunk_buffer['sequences']) >= CHUNK_SIZE:
                    save_chunk(chunk_buffer, file_basename, chunk_counter)
                    chunk_buffer = {'sequences': [], 'actions': []}
                    chunk_counter += 1
        
        frame_idx += 1
        
    if len(chunk_buffer['sequences']) > 0:
        save_chunk(chunk_buffer, file_basename, chunk_counter)
        
    cap.release()
    return local_collected

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    all_mp4 = glob.glob(os.path.join(DATA_DIR, "*.mp4"))
    
    # 构建任务列表，只取前 PROCESS_FILE_LIMIT 个视频
    tasks = []
    for mp4_f in all_mp4[:PROCESS_FILE_LIMIT]:
        osr_f = mp4_f.replace('.mp4', '.osr')
        if os.path.exists(osr_f):
            tasks.append((mp4_f, osr_f))
            
    print(f"Starting Multiprocessing with {MAX_WORKERS} workers on {len(tasks)} videos...")
    
    total_samples = 0
    # 使用 ProcessPoolExecutor 并发处理
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 使用 tqdm 显示进度条
        results = list(tqdm(executor.map(process_single_video, tasks), total=len(tasks)))
        
    total_samples = sum(results)
    print(f"\n[Done] Total samples collected: {total_samples}")
