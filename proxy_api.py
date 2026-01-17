import requests
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from concurrent.futures import ThreadPoolExecutor

# --- 配置部分 ---
CLASH_API_URL = "http://127.0.0.1:62475"
API_SECRET = "f57aca82-8ce9-4f55-9606-c45216435229"  # 如果 config.yaml 设置了 secret，请填入
TARGET_GROUP = "🚀 节点选择"  # 你想要自动切换的策略组名称
TEST_URL = "http://www.gstatic.com/generate_204"
TIMEOUT_MS = 2000

# FastAPI 实例
app = FastAPI(title="Clash Auto Switcher")

# 构建 Header
headers = {"Content-Type": "application/json"}
if API_SECRET:
    headers["Authorization"] = f"Bearer {API_SECRET}"

# --- 核心功能函数 ---

def get_proxies():
    """获取所有代理信息"""
    try:
        url = f"{CLASH_API_URL}/proxies"
        resp = requests.get(url, headers=headers, timeout=3)
        resp.raise_for_status()
        return resp.json()['proxies']
    except Exception as e:
        print(f"[Error] Clash API 连接失败: {e}")
        return None

def test_latency(proxy_name):
    """测试单个节点延迟"""
    safe_name = requests.utils.quote(proxy_name)
    url = f"{CLASH_API_URL}/proxies/{safe_name}/delay"
    params = {"timeout": TIMEOUT_MS, "url": TEST_URL}
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=3)
        if resp.status_code == 200:
            return proxy_name, resp.json().get('delay', 99999)
    except:
        pass
    return proxy_name, -1

def switch_proxy_request(selector_name, node_name):
    """发送切换请求"""
    url = f"{CLASH_API_URL}/proxies/{selector_name}"
    payload = json.dumps({"name": node_name})
    try:
        resp = requests.put(url, headers=headers, data=payload, timeout=3)
        return resp.status_code == 204
    except:
        return False

# --- API 路由 ---

import random  # ← 在文件顶部导入

@app.get("/switch")
def trigger_switch():
    """随机切换到一个可用节点"""
    all_proxies = get_proxies()
    if not all_proxies:
        raise HTTPException(status_code=503, detail="无法连接到 Clash API")
    
    if TARGET_GROUP not in all_proxies:
        raise HTTPException(status_code=404, detail=f"策略组 '{TARGET_GROUP}' 不存在")

    # 1. 直接筛选有效节点
    candidates = all_proxies[TARGET_GROUP]['all']
    valid_candidates = [n for n in candidates if n not in ["DIRECT", "REJECT", "REJECT", "RecycleBin"]]
    
    if not valid_candidates:
        raise HTTPException(status_code=404, detail="该策略组下没有有效节点")

    # 2. 随机选择一个！不搞延迟测试那一套
    selected_node = random.choice(valid_candidates)
    
    # 3. 执行切换
    current_node = all_proxies[TARGET_GROUP].get('now')
    switched = False
    
    if current_node != selected_node:
        success = switch_proxy_request(TARGET_GROUP, selected_node)
        if success:
            switched = True
        else:
            raise HTTPException(status_code=500, detail="切换请求发送失败")
    
    return {
        "status": "success",
        "action": "switched" if switched else "kept",
        "group": TARGET_GROUP,
        "previous_node": current_node,
        "current_node": selected_node,
        "selected_randomly": True,
        "candidates_count": len(valid_candidates)
    }

# --- 启动入口 ---
if __name__ == "__main__":
    # host="0.0.0.0" 允许局域网访问，方便手机控制
    uvicorn.run(app, host="0.0.0.0", port=8000)