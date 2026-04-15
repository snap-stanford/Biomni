#!/usr/bin/env python3
"""
调试脚本：向工具后端发送测试请求并打印结果
============================================
用法：
    python send_request_template.py

依赖：
    pip install requests

配置：
    修改下方 WORKER_IP / TOOL_NAME / ACTION / PAYLOAD 四个变量即可。
"""

import time

import requests

# ============================================================
# 配置区（按需修改）
# ============================================================
WORKER_IP = "100.103.118.72"  # 工具后端服务器 IP
PORT = 8001

TOOL_NAME = "test_tool"  # 对应 tools/<TOOL_NAME>/ 目录名
ACTION = "default"  # config.json 中定义的 action 名；单脚本工具填 "default"

# 发送给工具脚本的参数（会被写入 params.json）
PAYLOAD = {
    "smiles_list": [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # 阿司匹林
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # 咖啡因
        "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",  # 青霉素G
    ]
}

TIMEOUT_SECS = 300  # 最长等待时间（秒）
POLL_INTERVAL = 3  # 轮询间隔（秒）
# ============================================================


BASE_URL = f"http://{WORKER_IP}:{PORT}"
RUN_URL = f"{BASE_URL}/run/{TOOL_NAME}/{ACTION}"
JOB_URL = f"{BASE_URL}/job"

# 跳过系统代理，防止内网 IP 被发往外网代理导致 502
NO_PROXY = {"http": None, "https": None}


def submit_job(payload: dict) -> str:
    """提交任务，返回 job_id。"""
    print(f"[1/3] 提交任务到 {RUN_URL}")
    r = requests.post(RUN_URL, json=payload, timeout=10, proxies=NO_PROXY)
    r.raise_for_status()
    data = r.json()

    if "error" in data:
        raise RuntimeError(f"任务提交失败: {data['error']}")

    job_id = data["job_id"]
    print(f"      Job ID: {job_id}")
    return job_id


def poll_job(job_id: str) -> dict:
    """轮询任务状态，直到 finished / failed / 超时。"""
    print(f"[2/3] 等待任务完成（最长 {TIMEOUT_SECS}s）…")
    start = time.time()

    while True:
        elapsed = time.time() - start
        if elapsed > TIMEOUT_SECS:
            raise TimeoutError(f"超时：任务 {job_id} 在 {TIMEOUT_SECS}s 内未完成")

        try:
            r = requests.get(f"{JOB_URL}/{job_id}", timeout=10, proxies=NO_PROXY)
            status = r.json()
        except Exception as e:
            print(f"      [WARN] 查询失败，稍后重试: {e}", flush=True)
            time.sleep(POLL_INTERVAL)
            continue

        state = status.get("status")
        print(f"      [{elapsed:5.1f}s] 状态: {state}", flush=True)

        if state == "running":
            time.sleep(POLL_INTERVAL)
            continue
        elif state == "failed":
            raise RuntimeError(f"服务器崩溃: {status.get('data')}")
        elif state == "finished":
            return status.get("data") or {}
        else:
            # 未知状态，等一会儿再试
            time.sleep(POLL_INTERVAL)


def print_result(result: dict):
    """格式化打印结果。"""
    print("\n[3/3] 结果:")
    if not result.get("success"):
        print(f"  ❌ 工具内部错误: {result.get('error')}")
        return

    # 打印 summary
    summary = result.get("summary", {})
    if summary:
        print("  --- Summary ---")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    # 打印 results（截断显示，避免刷屏）
    results = result.get("results")
    if isinstance(results, list):
        print(f"\n  --- Results (前 5 条 / 共 {len(results)} 条) ---")
        for item in results[:5]:
            print(f"  {item}")
    elif isinstance(results, dict):
        print("\n  --- Results ---")
        # 字典形式：每个 key 最多显示 5 条
        for key, val in results.items():
            if isinstance(val, list):
                print(f"  {key} (前 5 条):")
                for item in val[:5]:
                    print(f"    {item}")
            else:
                print(f"  {key}: {val}")

    errors = result.get("errors")
    if errors:
        print(f"\n  ⚠️  部分错误: {errors}")

    print("\n  ✅ 测试成功！")


def main():
    print("=" * 50)
    print(f"工具: {TOOL_NAME} / action: {ACTION}")
    print(f"后端: {BASE_URL}")
    print("=" * 50)

    try:
        job_id = submit_job(PAYLOAD)
        result = poll_job(job_id)
        print_result(result)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")


if __name__ == "__main__":
    main()
