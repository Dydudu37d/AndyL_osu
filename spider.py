import requests
import json
import time
import os
from bs4 import BeautifulSoup

# ================= 配置区域 =================
# 务必从你的浏览器 F12 网络请求中复制这两个值
COOKIES = {
    'osu_session': 'eyJpdiI6IkpvUWd6NUNDcmV6TWZFV3dQTm16THc9PSIsInZhbHVlIjoiS0trMU9aS1ZoNk56Z0o0NC9RMUx0b2FGemhsd1Fkd1RTQU53eXhUd01RZ1hVMDBHWnNzUU9pdUdBYVRxazlTNlFvajZHNEVpbXA4NXRialpqa1YxNi80UE5NOHpEUFJjZjFEbThsRUkvVm9JWjFmNHRzYW9udnFzMEFYTHVEbmJQOHY0UXozYUdmVGRoZko2RC9MdCt3PT0iLCJtYWMiOiIxYWQ1ZmYyZDU3MWI0MmQzYjBjOGY2MWNhNzNhYjI2MzI0NGY2Nzg0MTE5ZGQzODZiY2VlYmE2YzBlZWU1ZDUyIiwidGFnIjoiIn0%3D', 
    
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://osu.ppy.sh/beatmapsets',
    'X-Requested-With': 'XMLHttpRequest' # 伪装成 AJAX 请求
}

SAVE_DIR = "./osr_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

# 延时设置 (防止被Cloudflare 5秒盾拦截或封IP)
DELAY_BETWEEN_MAPS = 2.0
DELAY_BETWEEN_DOWNLOADS = 1.5
# ===========================================

def get_beatmap_list(cursor_string=None):
    """获取谱面列表 (解析 HTML 中的 JSON)"""
    url = "https://osu.ppy.sh/beatmapsets"
    params = {
        'm': 0, # osu! standard
        's': 'ranked', # 只爬 Ranked
        'sort': 'plays_desc' # 按游玩次数排序，保证有回放
    }
    if cursor_string:
        params['cursor_string'] = cursor_string

    try:
        # 这里不能直接请求API，而是请求网页
        r = requests.get(url, params=params, headers=HEADERS, cookies=COOKIES)
        r.raise_for_status()
        
        soup = BeautifulSoup(r.text, 'html.parser')
        script = soup.find('script', {'id': 'json-beatmaps'})
        
        if not script:
            print("[-] 未找到 beatmap 数据，可能 Cookie 过期或触发了 CF 验证")
            return None, None

        data = json.loads(script.string)
        next_cursor = data.get('cursor_string')
        beatmapsets = data.get('beatmapsets', [])
        return beatmapsets, next_cursor
        
    except Exception as e:
        print(f"[-] 获取谱面列表失败: {e}")
        return None, None

def get_scores(beatmap_id):
    """获取单张谱面的分数榜 (直接 API)"""
    url = f"https://osu.ppy.sh/beatmaps/{beatmap_id}/scores"
    params = {'mode': 'osu', 'type': 'global'}
    
    try:
        r = requests.get(url, params=params, headers=HEADERS, cookies=COOKIES)
        
        # 如果也是 HTML 返回，说明可能被重定向了，通常是 Cookie 失效
        if 'text/html' in r.headers.get('Content-Type', ''):
            print("[-] 获取分数返回了 HTML，Cookie 可能失效")
            return []
            
        data = r.json()
        return data.get('scores', [])
    except Exception as e:
        print(f"[-] 获取分数失败 map_id={beatmap_id}: {e}")
        return []

def download_replay(score_id, map_id):
    url = f"https://osu.ppy.sh/scores/{score_id}/download"
    filename = os.path.join(SAVE_DIR, f"{map_id}_{score_id}.osr")
    
    if os.path.exists(filename):
        print(f"[*] 文件已存在: {filename}")
        return

    try:
        # stream=True 保持流式下载
        r = requests.get(url, headers=HEADERS, cookies=COOKIES, stream=True)
        
        # 如果状态码不是 200，直接报错
        if r.status_code != 200:
            print(f"[-] 下载失败 {score_id}: HTTP {r.status_code}")
            return

        # === 修改点：不管有没有 Content-Length，先写入文件 ===
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # === 验货阶段：检查下载下来的文件实际有多大 ===
        file_size = os.path.getsize(filename)
        
        if file_size < 1000: # 如果落地文件小于 1KB，那肯定是报错页面
            print(f"[!] 文件无效 (仅 {file_size} bytes)，已删除: {score_id}")
            os.remove(filename) # 删掉这个垃圾文件
            
            # 调试：看看里面写了啥（通常是 HTML 报错）
            # with open(filename, 'r', errors='ignore') as f:
            #     print(f"    内容预览: {f.read()[:100]}")
        else:
            print(f"[+] 下载成功: {filename} ({file_size / 1024:.2f} KB)")
            
    except Exception as e:
        print(f"[-] 下载异常 {score_id}: {e}")
        # 如果下载中途崩了，把半成品删掉
        if os.path.exists(filename):
            os.remove(filename)

def main():
    print("=== 开始爬取 osu! 回放数据 ===")
    cursor = None
    
    # 循环爬取每一页的谱面集
    while True:
        print(f"\n[*] 正在获取谱面列表页 (Cursor: {cursor})...")
        beatmapsets, cursor = get_beatmap_list(cursor)
        
        if not beatmapsets:
            print("[-] 无法获取谱面列表，停止。")
            break
            
        print(f"[*] 本页获取到 {len(beatmapsets)} 个谱面集")
        
        for bset in beatmapsets:
            # 遍历该 Set 下的所有难度
            for beatmap in bset.get('beatmaps', []):
                # 只爬 osu! 模式 (mode_int = 0)
                if beatmap['mode_int'] != 0:
                    continue
                
                bid = beatmap['id']
                difficulty = beatmap['difficulty_rating']
                
                # 过滤：如果你只想要高星图 (比如 5星以上)
                if difficulty < 5.0: 
                    continue

                print(f"  -> 分析谱面: {bid} ({beatmap['version']} - {difficulty}*)")
                
                # 获取该谱面的前 50 名分数
                scores = get_scores(bid)
                
                count = 0
                for score in scores:
                    sid = score['id']
                    
                    # 可以在这里加过滤逻辑，例如：
                    # if not score['passed']: continue 
                    # if 'DT' not in score['mods']: continue 
                    
                    download_replay(sid, bid)
                    count += 1
                    time.sleep(DELAY_BETWEEN_DOWNLOADS) # 关键：防止封号
                    
                    # 演示用：每个图只下前 3 个 replay，避免你测试时下太久
                    if count >= 3: 
                        break 
                
                time.sleep(DELAY_BETWEEN_MAPS)
        
        if not cursor:
            print("[*] 所有页面遍历完成")
            break
            
        time.sleep(5) # 翻页间隔

if __name__ == "__main__":
    main()