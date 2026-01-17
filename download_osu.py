import os
import re
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

# ================= 配置区域 =================
OSR_DIR = "./dataset_clean/NM_HD"    # 你的 .osr 文件夹路径
DOWNLOAD_DIR = "./songs_download"    # 铺面下载保存路径

# 线程数 (osu! 官网限制较严，建议不要超过 4-6)
MAX_WORKERS = 4 

# 把你提供的 Header 完整填在这里
HEADERS = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'cache-control': 'no-cache',
    'pragma': 'no-cache',
    'priority': 'u=0, i',
    'referer': 'https://osu.ppy.sh/beatmapsets/', # 动态填充
    'sec-ch-ua': '"Microsoft Edge";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0',
    # === 这里的 Cookie 务必保持最新 ===
    'cookie': 'locale=zh; osu_session=eyJpdiI6IjRLaFE1UE4wYkJpbW0ydkIxcHZYS1E9PSIsInZhbHVlIjoiT1Y1V09Ud0duSWFMRzFHcWVSREFKYlJXQkpBSGVDWWdiaW1QcnhGbzJaVE5iM0VES1d3SGFuK0VVSm50V1VIbitLbHcydExEVEVzQ3lDZTJwMyt3WjhjREhYYXp3aEhHaDBGT3QyZUNLdUVwZnBPOUdaTG52bW5KL21uclN1TWRmS0JvWnJJSlBZemhScm1vbURWMURBPT0iLCJtYWMiOiI1MGRiMTdiOGNlODBiZTIyZTViYmExNTNlNWUzNWViNmUwYjBmOGI3YThiY2ExMTgyZjg3YjQxMzYwMTQ5MDI1IiwidGFnIjoiIn0%3D'
}
# ===========================================

if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

def extract_beatmap_ids():
    """第一步：从文件名提取 Beatmap ID"""
    ids = set()
    print(f"[*] 正在扫描 {OSR_DIR} ...")
    for f in os.listdir(OSR_DIR):
        if f.endswith(".osr"):
            # 匹配 123456_xxxxx.osr
            m = re.match(r"^(\d+)_", f)
            if m:
                ids.add(m.group(1))
    print(f"[+] 找到 {len(ids)} 个唯一的 Beatmap ID")
    return list(ids)

def resolve_set_id(beatmap_id, session):
    """
    第二步：Beatmap ID -> Set ID
    利用官网的 /b/{id} 跳转机制获取 Set ID
    """
    url = f"https://osu.ppy.sh/b/{beatmap_id}"
    try:
        # 禁止自动跳转，手动分析 Location，速度更快
        r = session.head(url, headers=HEADERS, allow_redirects=False, timeout=10)
        
        if r.status_code in [301, 302]:
            redirect_url = r.headers.get('Location', '')
            # 跳转格式通常是 https://osu.ppy.sh/beatmapsets/{set_id}#osu/{map_id}
            m = re.search(r"beatmapsets/(\d+)", redirect_url)
            if m:
                return m.group(1)
        
        # 如果没有跳转，可能直接返回了页面（或者 ID 无效）
        if r.status_code == 200:
             print(f"[-] ID {beatmap_id} 未发生跳转，可能已删除或未登录")
             return None

    except Exception as e:
        print(f"[-] 解析 ID {beatmap_id} 失败: {e}")
    return None

def download_file(set_id, session):
    """
    第三步：下载 (支持断点续传)
    """
    url = f"https://osu.ppy.sh/beatmapsets/{set_id}/download"
    file_path = os.path.join(DOWNLOAD_DIR, f"{set_id}.osz")
    temp_path = file_path + ".part"

    # 1. 检查是否已完成下载
    if os.path.exists(file_path):
        print(f"[skip] {set_id}.osz 已存在")
        return

    # 2. 检查是否有临时文件（断点续传）
    resume_byte_pos = 0
    if os.path.exists(temp_path):
        resume_byte_pos = os.path.getsize(temp_path)
    
    # 设置 Range 头
    download_headers = HEADERS.copy()
    download_headers['referer'] = f"https://osu.ppy.sh/beatmapsets/{set_id}"
    if resume_byte_pos > 0:
        download_headers['Range'] = f"bytes={resume_byte_pos}-"
        print(f"[resume] 继续下载 {set_id} (从 {resume_byte_pos/1024:.1f}KB 开始)...")
    else:
        print(f"[start] 开始下载 {set_id} ...")

    try:
        # 发起请求 (允许重定向，因为下载链接会 302 到 CDN)
        r = session.get(url, headers=download_headers, stream=True, timeout=20, allow_redirects=True)
        
        # 检查是否因为需要登录被重定向到了登录页 (Login Check)
        if "users/login" in r.url:
            print(f"[!] Cookie 失效，被重定向到登录页！请更新 Cookie。")
            return

        mode = 'ab' if resume_byte_pos > 0 else 'wb'
        
        # 404 或 403 处理
        if r.status_code in [403, 404]:
            print(f"[-] 下载失败 {set_id}: HTTP {r.status_code} (可能是需付费用户下载或图被删)")
            if os.path.exists(temp_path): os.remove(temp_path)
            return
            
        # 416 Range Not Satisfiable (说明本地文件比服务器的大，或者已经下完了)
        if r.status_code == 416:
            print(f"[!] Range 错误，重新下载 {set_id}")
            resume_byte_pos = 0
            mode = 'wb'
            download_headers.pop('Range', None)
            r = session.get(url, headers=download_headers, stream=True, timeout=20)

        total_size = int(r.headers.get('content-length', 0))
        
        with open(temp_path, mode) as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # 下载完成，重命名
        os.rename(temp_path, file_path)
        print(f"[ok] 下载完成: {set_id}.osz")

    except Exception as e:
        print(f"[err] 下载异常 {set_id}: {e}")

def worker(beatmap_id, session):
    """
    工作线程：负责 解析 -> 下载 的全流程
    """
    # 1. 转换 ID
    set_id = resolve_set_id(beatmap_id, session)
    if not set_id:
        # print(f"[-] 无法获取 Set ID: {beatmap_id}")
        return

    # 2. 执行下载
    download_file(set_id, session)

def main():
    # 1. 获取所有 ID
    beatmap_ids = extract_beatmap_ids()
    if not beatmap_ids:
        return

    # 2. 启动线程池
    print(f"[*] 启动 {MAX_WORKERS} 个线程开始处理...")
    
    # 使用 session 保持连接池，提高效率
    session = requests.Session()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(worker, bid, session) for bid in beatmap_ids]
        
        # 等待完成
        for future in as_completed(futures):
            pass # 可以在这里处理异常，但 worker 内部已经打印了

    print("\n=== 所有任务处理完毕 ===")

if __name__ == "__main__":
    main()