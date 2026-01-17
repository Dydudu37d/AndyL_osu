import json
import os
import requests
import time

# --- 配置区域 ---
JSON_FILE = 'render_tasks.json'  # 你的JSON文件名
# 你的cURL headers配置
HEADERS = {
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'cache-control': 'no-cache',
    'origin': 'https://ordr.issou.best',
    'pragma': 'no-cache',
    'priority': 'u=1, i',
    'referer': 'https://ordr.issou.best/',
    'sec-ch-ua': '"Microsoft Edge";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0'
}
# ----------------

def download_missing_videos():
    # 1. 读取 JSON 文件
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {JSON_FILE}")
        return
    except json.JSONDecodeError:
        print(f"错误: {JSON_FILE} 格式不正确")
        return

    print(f"共加载 {len(data)} 条记录，开始检查...")

    for item in data:
        # 获取必要字段
        output_path = item.get('output_path')
        render_id = item.get('render_id')
        
        if not output_path or not render_id:
            continue

        # 规范化路径 (处理混合斜杠问题)
        output_path = os.path.normpath(output_path)

        # 2. 检查文件是否存在
        if os.path.exists(output_path):
            # 文件存在，跳过
            print(f"[已存在] ID: {render_id} | Path: {output_path}")
            continue
        
        # 文件不存在，开始处理
        print(f"\n[缺失] 正在处理 ID: {render_id}...")
        
        # 3. 请求 API 获取下载链接
        api_url = f"https://apis.issou.best/dynlink/ordr/gen?id={render_id}"
        
        try:
            # 获取下载直链
            response = requests.get(api_url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            download_url = result.get('url')
            
            if not download_url:
                print(f"  -> 错误: API 未返回 URL。响应: {result}")
                continue
                
            print(f"  -> 获取链接成功: {download_url}")
            
            # 4. 下载视频
            print(f"  -> 正在下载到: {output_path}")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 流式下载避免大文件占用过多内存
            video_resp = requests.get(download_url, stream=True, timeout=30)
            video_resp.raise_for_status()
            
            with open(output_path, 'wb') as v_file:
                for chunk in video_resp.iter_content(chunk_size=8192):
                    v_file.write(chunk)
            
            print("  -> 下载完成！")
            
            # 礼貌性延时，避免触发服务端的频率限制
            time.sleep(1)

        except Exception as e:
            print(f"  -> 处理失败: {e}")

if __name__ == "__main__":
    download_missing_videos()