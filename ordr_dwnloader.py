import os
import time
import json
import requests
import urllib3
from typing import List, Dict, Optional

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= 配置区域 =================
SOURCE_DIR = "./dataset_clean/NM_HD"
OUTPUT_DIR = "./ordr_videos_batch"
PROXY_SWITCH_URL = "http://127.0.0.1:8000/switch"
MAX_PROXY_SWITCHES = 2000
TASKS_SAVE_FILE = "render_tasks.json"

# 重试配置
DOWNLOAD_MAX_RETRIES = 20
DOWNLOAD_RETRY_DELAY = 5

# 新增：批量大小配置
BATCH_SIZE = 50

# 请求头和渲染参数保持不变...
HEADERS = {
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'cache-control': 'no-cache',
    'origin': 'https://ordr.issou.best',
    'pragma': 'no-cache',
    'referer': 'https://ordr.issou.best/',
    'sec-ch-ua': '"Microsoft Edge";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0'
}

RENDER_BODY = {
    'resolution': '1280x720',
    'globalVolume': '50',
    'musicVolume': '50',
    'hitsoundVolume': '50',
    'useSkinHitsounds': 'false',
    'playNightcoreSamples': 'true',
    'showHitErrorMeter': 'true',
    'showScore': 'true',
    'showHPBar': 'true',
    'showComboCounter': 'true',
    'showPPCounter': 'false',
    'showKeyOverlay': 'false',
    'showScoreboard': 'true',
    'showAvatarsOnScoreboard': 'false',
    'showBorders': 'false',
    'showMods': 'true',
    'showResultScreen': 'true',
    'showHitCounter': 'false',
    'showSliderBreaks': 'false',
    'showAimErrorMeter': 'false',
    'showStrainGraph': 'false',
    'customSkin': 'false',
    'skin': 'osu_default_skin_improved',
    'useSkinCursor': 'true',
    'useSkinColors': 'false',
    'useBeatmapColors': 'true',
    'cursorScaleToCS': 'false',
    'cursorRainbow': 'false',
    'cursorTrailGlow': 'false',
    'drawFollowPoints': 'true',
    'scaleToTheBeat': 'false',
    'sliderMerge': 'false',
    'objectsRainbow': 'false',
    'objectsFlashToTheBeat': 'false',
    'useHitCircleColor': 'true',
    'seizureWarning': 'false',
    'loadStoryboard': 'true',
    'loadVideo': 'true',
    'introBGDim': '0',
    'inGameBGDim': '75',
    'breakBGDim': '30',
    'BGParallax': 'false',
    'showDanserLogo': 'false',
    'skip': 'true',
    'cursorRipples': 'false',
    'sliderSnakingIn': 'true',
    'sliderSnakingOut': 'true',
    'cursorSize': '1',
    'cursorTrail': 'true',
    'showUnstableRate': 'true',
    'drawComboNumbers': 'true',
    'addPitch': 'false',
    'noDelete': 'false',
    'turboMode': 'false',
    'ignoreFail': 'false',
    'aimErrorMeterXPos': '1222',
    'aimErrorMeterYPos': '586',
    'ppCounterXPos': '5',
    'ppCounterYPos': '150',
    'hitCounterXPos': '5',
    'hitCounterYPos': '195',
    'strainGraphXPos': '5',
    'strainGraphYPos': '240'
}

# ===========================================

class TaskManager:
    """管理任务状态，支持断点续传"""
    def __init__(self, save_file: str):
        self.save_file = save_file
        self.tasks: List[Dict] = self.load_tasks()
    
    def load_tasks(self) -> List[Dict]:
        """从文件加载任务列表"""
        if os.path.exists(self.save_file):
            with open(self.save_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save_tasks(self):
        """保存任务列表到文件"""
        with open(self.save_file, 'w', encoding='utf-8') as f:
            json.dump(self.tasks, f, indent=2, ensure_ascii=False)
        print(f"[*] 任务状态已保存 ({len(self.tasks)} 个任务)")
    
    def get_pending_uploads(self) -> List[str]:
        """获取待上传的文件列表"""
        uploaded_files = {task['filename'] for task in self.tasks}
        all_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".osr")]
        return [f for f in all_files if f not in uploaded_files]
    
    def add_task(self, filename: str, render_id: str):
        """添加新任务"""
        self.tasks.append({
            "filename": filename,
            "render_id": render_id,
            "status": "pending",
            "created_at": time.time(),
            "output_path": os.path.join(OUTPUT_DIR, filename.replace(".osr", ".mp4")),
            "download_retries": 0
        })
        self.save_tasks()
    
    def get_active_tasks(self) -> List[Dict]:
        """获取需要轮询的任务（pending/rendering）"""
        return [t for t in self.tasks if t['status'] in ['pending', 'rendering']]
    
    def update_task_status(self, render_id: str, status: str, error_code: Optional[str] = None):
        """更新任务状态"""
        for task in self.tasks:
            if task['render_id'] == render_id:
                task['status'] = status
                if error_code:
                    task['error_code'] = error_code
                self.save_tasks()
                break
    
    def mark_task_done(self, render_id: str):
        """标记任务完成"""
        self.update_task_status(render_id, "done")
    
    def mark_task_error(self, render_id: str, error_code: str):
        """标记任务失败"""
        self.update_task_status(render_id, "error", error_code)
    
    def increment_download_retry(self, render_id: str) -> bool:
        """增加下载重试计数，返回是否还能继续重试"""
        for task in self.tasks:
            if task['render_id'] == render_id:
                task['download_retries'] = task.get('download_retries', 0) + 1
                can_retry = task['download_retries'] <= DOWNLOAD_MAX_RETRIES
                if not can_retry:
                    task['status'] = 'error'
                self.save_tasks()
                return can_retry
        return False

def switch_proxy() -> bool:
    """切换IP代理（忽略SSL）"""
    print(f"[*] 尝试切换IP代理...")
    try:
        r = requests.get(PROXY_SWITCH_URL, timeout=10, verify=False)
        if r.status_code == 200 and r.json().get('status') == 'success':
            print(f"[+] IP切换成功: {r.json().get('current_node', '未知节点')}")
            return True
        print(f"[-] IP切换失败: HTTP {r.status_code}")
        return False
    except Exception as e:
        print(f"[-] IP切换异常: {e}")
        return False

def batch_upload(task_manager: TaskManager) -> bool:
    """
    批量上传所有.osr文件
    返回: True 表示还有未上传的文件，False 表示所有文件已上传
    """
    pending = task_manager.get_pending_uploads()
    
    if not pending:
        print("[*] 所有文件已上传过，跳过提交阶段")
        return False
    
    print(f"=== 批量提交阶段: 共 {len(pending)} 个文件待上传 ===")
    print(f"IP切换策略: 启用 (每个文件最多{MAX_PROXY_SWITCHES}次)")

    consecutive_429_count = 0
    
    for idx, filename in enumerate(pending):
        # 检查活跃任务数，达到BATCH_SIZE时立即中断
        active_count = len(task_manager.get_active_tasks())
        if active_count >= BATCH_SIZE:
            print(f"\n[!] 活跃任务数达到 {active_count}，暂停上传，开始轮询...")
            return True
        
        print(f"\n[{idx+1}/{len(pending)}] {filename}")
        
        # 检查视频是否已存在（双重保险）
        output_mp4 = os.path.join(OUTPUT_DIR, filename.replace(".osr", ".mp4"))
        if os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 1024:
            print("[Skip] 视频已存在")
            continue
        
        render_id = upload_replay(
            os.path.join(SOURCE_DIR, filename), 
            max_switches=MAX_PROXY_SWITCHES
        )
        
        if render_id:
            task_manager.add_task(filename, render_id)
            consecutive_429_count = 0
        else:
            consecutive_429_count += 1
            print(f"[-] 失败，连续失败次数: {consecutive_429_count}")
            
            if consecutive_429_count >= 5:
                print("[!] 警告：连续5次提交失败，可能所有IP均被限制")
                print("[!] 停止提交，进入轮询下载阶段...")
                break
    
    print("\n[*] 批量提交完成")
    return len(task_manager.get_pending_uploads()) > 0

def upload_replay(file_path: str, max_switches: int = MAX_PROXY_SWITCHES) -> Optional[str]:
    """上传单个文件，返回render_id（忽略SSL）"""
    url = 'https://apis.issou.best/ordr/renders'
    filename = os.path.basename(file_path)
    print(f"[*] 上传: {filename}")
    
    attempt = 0
    proxy_switch_count = 0
    
    while attempt < 10:
        try:
            with open(file_path, 'rb') as f:
                files = {'replayFile': (filename, f, 'application/octet-stream')}
                r = requests.post(url, headers=HEADERS, data=RENDER_BODY, files=files, timeout=30, verify=False)
            
            if r.status_code in [200, 201]:
                render_id = r.json().get('renderID')
                if render_id is not None:
                    render_id = str(render_id)
                print(f"[+] 成功: RenderID={render_id[:8]}...")
                return render_id
            
            elif r.status_code == 429:
                print(f"[-] 429限流触发")
                
                if proxy_switch_count < max_switches:
                    if switch_proxy():
                        proxy_switch_count += 1
                        time.sleep(2)
                        continue
                    print("-> IP切换失败")
                
                print(f"-> 等待60秒... (重试={attempt+1}/10)")
                time.sleep(60)
                attempt += 1
                continue
            
            else:
                print(f"[-] 上传失败: HTTP {r.status_code}")
                return None
        
        except requests.exceptions.ProxyError as e:
            print(f"[-] 代理连接异常: {e}")
            if proxy_switch_count < max_switches:
                print(f"[*] 检测到代理错误，尝试切换代理 ({proxy_switch_count + 1}/{max_switches})...")
                if switch_proxy():
                    proxy_switch_count += 1
                    time.sleep(2)
                    continue
            
            print("[-] 代理切换次数耗尽或失败，等待重试...")
            attempt += 1
            time.sleep(5)
            
        except Exception as e:
            print(f"[-] 上传异常: {e}")
            attempt += 1
            time.sleep(5)
    
    print("[-] 重试次数耗尽，标记为失败")
    return None

def poll_all_tasks(task_manager: TaskManager):
    """集中轮询所有活跃任务（修复ID类型问题）"""
    url = 'https://apis.issou.best/ordr/rendersweb'
    poll_interval = 10
    
    print("\n=== 集中轮询阶段 ===")
    
    while True:
        active_tasks = task_manager.get_active_tasks()
        
        if not active_tasks:
            print("\n[✓] 所有任务已完成或失败")
            break
        
        print(f"\n[*] 活跃任务: {len(active_tasks)} 个")
        
        # 批量查询优化：减少请求次数
        for task in active_tasks:
            render_id = task['render_id']
            filename_display = task['filename'][:35]
            
            try:
                print(f"[*] 查询: {filename_display:<35} (ID: {render_id[:8]}...)")
                
                params = {'renderID': render_id, 'pageSize': 1, 'page': 1}
                r = requests.get(url, headers=HEADERS, params=params, timeout=10, verify=False)
                data = r.json()
                
                # 更健壮的解析
                if not isinstance(data, dict) or 'renders' not in data:
                    print(f"    [-] 无效响应格式")
                    continue
                
                renders = data['renders']
                if not isinstance(renders, list) or len(renders) == 0:
                    print(f"    [-] 无渲染记录")
                    continue
                
                item = renders[0]
                if not isinstance(item, dict):
                    print(f"    [-] 无效记录格式")
                    continue
                
                # 关键修复：将API返回的ID转换为字符串
                api_render_id = str(item.get('renderID', ''))
                status = item.get('progress', 'Unknown')
                
                # 比较时也使用字符串
                if api_render_id == render_id:
                    print(f"    {filename_display:<35} -> {status}")
                    
                    if "Done" in status:
                        print(f"[+] 完成: {task['filename']}")
                        download_and_cleanup(task_manager, task)
                    elif "Error" in status or "Failed" in status:
                        print(f"[-] 失败: {task['filename']} (错误码: {item.get('errorCode', 'Unknown')})")
                        task_manager.mark_task_error(render_id, item.get('errorCode'))
                    else:
                        # 渲染中，更新状态
                        task_manager.update_task_status(render_id, "rendering")
                else:
                    print(f"    [-] ID不匹配: 期望 {render_id[:8]}..., 得到 {api_render_id[:8]}...")
                
            except requests.exceptions.SSLError as e:
                print(f"    [!] SSL错误 (已忽略): {render_id[:8]}...")
                time.sleep(2)
                continue
            
            except Exception as e:
                print(f"    [-] 查询异常: {str(e)[:50]}")
                time.sleep(5)
                continue
        
        print(f"[Sleep] {poll_interval}秒后继续轮询...")
        time.sleep(poll_interval)

def download_and_cleanup(task_manager: TaskManager, task: Dict):
    """下载视频并从活跃列表移除（带重试机制）"""
    url = f'https://apis.issou.best/dynlink/ordr/gen?id={task["render_id"]}'
    
    try:
        r = requests.get(url, headers=HEADERS, timeout=10, verify=False)
        d_url = r.json().get('url')
        if not d_url:
            print(f"[-] 无法获取下载链接")
            return False
    except Exception as e:
        print(f"[-] 获取链接异常: {e}")
        return False
    
    success = download_video(d_url, task['output_path'], task['render_id'], task_manager)
    
    if success:
        task_manager.mark_task_done(task['render_id'])
        return True
    else:
        if task_manager.increment_download_retry(task['render_id']):
            print(f"[*] 将在下次轮询时重试下载 (尝试 {task['download_retries']}/{DOWNLOAD_MAX_RETRIES})")
        else:
            print(f"[-] 下载重试次数耗尽，标记为失败")
        return False

def download_video(url: str, output_path: str, render_id: str = None, task_manager: TaskManager = None) -> bool:
    """下载视频（支持断点续传和重试，忽略SSL）"""
    if os.path.exists(output_path):
        existing_size = os.path.getsize(output_path)
        if existing_size > 1024:
            print(f"[Skip] 文件已存在: {output_path}")
            return True
    else:
        existing_size = 0
    
    print(f"[*] 下载中: {os.path.basename(output_path)}")
    
    try:
        headers = {'Range': f'bytes={existing_size}-'} if existing_size > 0 else {}
        headers.update(HEADERS)
        
        with requests.get(url, headers=headers, stream=True, timeout=60, verify=False) as r:
            r.raise_for_status()
            mode = 'ab' if existing_size > 0 else 'wb'
            with open(output_path, mode) as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"[OK] 下载完成: {output_path}")
        return True
        
    except requests.exceptions.SSLError as e:
        print(f"[!] SSL错误 (已忽略): {os.path.basename(output_path)}")
        return True
    
    except Exception as e:
        if hasattr(e, 'response') and e.response and e.response.status_code == 416:
            print(f"[OK] 文件已完整: {output_path}")
            return True
        
        print(f"[-] 下载失败: {e}")
        return False

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    task_manager = TaskManager(TASKS_SAVE_FILE)
    
    print(f"=== o!rdr 批量渲染器 (支持SSL忽略+下载重试+动态批处理) ===")
    print(f"批量大小: {BATCH_SIZE} (活跃任务数达到{BATCH_SIZE}时立即轮询)")
    
    # 循环处理：上传 -> 轮询 -> 上传...
    while True:
        # 阶段1：批量提交，直到任务数达到BATCH_SIZE或所有文件上传完成
        has_more_files = batch_upload(task_manager)
        
        # 阶段2：集中轮询
        poll_all_tasks(task_manager)
        
        # 如果所有文件已上传且没有活跃任务，退出循环
        if not has_more_files:
            pending_files = task_manager.get_pending_uploads()
            if not pending_files:
                print("\n[*] 所有文件已处理完成")
                break
    
    # 总结
    all_tasks = task_manager.tasks
    done = sum(1 for t in all_tasks if t['status'] == 'done')
    error = sum(1 for t in all_tasks if t['status'] == 'error')
    print(f"\n=== 任务总结 ===")
    print(f"总任务数: {len(all_tasks)}")
    print(f"成功: {done}")
    print(f"失败: {error}")

if __name__ == "__main__":
    main()