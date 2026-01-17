import os
import subprocess
import json
import platform
import shutil
import time

# ================= 配置区域 =================

# 指向你的 danser-cli 可执行文件 (如果是子目录，请带上目录名)
# 建议使用绝对路径，防止出错
DANSER_PATH = r"./danser/danser-cli.exe" 

# .osr 回放文件所在的文件夹
REPLAY_DIR = r"./dataset_clean/NM_HD"

# 最终视频保存目录
OUTPUT_DIR = r"./videos_output"

# FFmpeg 滤镜 (灰度 + 去音)
FFMPEG_FILTERS = "-c:v libx264 -pix_fmt yuv420p -preset fast -crf 23 -vf hue=s=0 -an"

# ===========================================

def render_replay(replay_path, output_filename):
    """
    调用 danser 生成视频 (修复路径 Bug 版)
    """
    # 最终想要的文件路径
    final_dest_path = os.path.join(os.path.abspath(OUTPUT_DIR), output_filename + ".mp4")

    if os.path.exists(final_dest_path):
        print(f"[跳过] 视频已存在: {output_filename}")
        return

    # 1. 准备 Danser 的工作环境
    danser_exe = os.path.abspath(DANSER_PATH)
    danser_work_dir = os.path.dirname(danser_exe) # 获取 danser 所在的文件夹
    
    # 2. 构造临时输出路径 (让 Danser 输出到它自己的 videos 文件夹)
    # 只要文件名，不要路径！
    temp_out_name = f"temp_{output_filename}" 
    
    # 构造 sPatch 配置
    patch_config = {
        "Recording": {
            "FrameWidth": 1280,
            "FrameHeight": 720,
            "ShowInterface": True,
            "EncoderOptions": FFMPEG_FILTERS,
            "OutputDir": "videos" # 强制确保它输出到 videos
        },
        "Audio": {
            "MasterVolume": 0.0
        },
        "General": {
            "DiscordPresenceOn": False
        }
    }
    patch_json = json.dumps(patch_config)

    # 3. 构造命令
    cmd = [
        danser_exe,
        "-r", os.path.abspath(replay_path),
        "-record",
        "-out", temp_out_name, # 关键修改：只传文件名，不传路径
        "-sPatch", patch_json,
        "-quickstart",
        "-noupdatecheck",
        "-nodbcheck",
        "-skip"
    ]

    print(f"[*] 正在渲染: {output_filename} ...")
    
    try:
        # cwd=danser_work_dir: 强制在 danser 目录下运行，确保能找到 assets
        # stdout=subprocess.DEVNULL: 隐藏刷屏日志，只看报错
        subprocess.run(cmd, cwd=danser_work_dir, check=True) # , stdout=subprocess.DEVNULL
        
        # 4. 搬运视频
        # Danser 默认生成的路径通常是: danser目录/videos/temp_文件名.mp4
        generated_file = os.path.join(danser_work_dir, "videos", temp_out_name + ".mp4")
        
        if os.path.exists(generated_file):
            print(f"[Move] 移动文件到目标目录...")
            shutil.move(generated_file, final_dest_path)
            print(f"[+] 成功: {final_dest_path}")
        else:
            print(f"[-] 错误: 渲染看似成功，但找不到生成的文件: {generated_file}")

    except subprocess.CalledProcessError as e:
        print(f"[-] 渲染进程崩溃: {replay_path}")
    except Exception as e:
        print(f"[-] 发生异常: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    if not os.path.exists(REPLAY_DIR):
        print(f"错误: 找不到回放目录 {REPLAY_DIR}")
        return

    files = [f for f in os.listdir(REPLAY_DIR) if f.endswith(".osr")]
    total = len(files)
    
    print(f"=== 开始批量渲染 (修复版): 共 {total} 个任务 ===")

    for idx, filename in enumerate(files):
        print(f"\n--- 进度 {idx+1}/{total} ---")
        replay_path = os.path.join(REPLAY_DIR, filename)
        output_name = os.path.splitext(filename)[0]
        
        render_replay(replay_path, output_name)

if __name__ == "__main__":
    main()