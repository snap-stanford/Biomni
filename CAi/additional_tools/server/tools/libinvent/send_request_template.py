#!/usr/bin/env python3
"""
调试脚本：向 libinvent 工具后端发送测试请求并打印结果
==================================================
用法：
    python send_request_template.py

依赖：
    pip install requests

说明：
- 本脚本风格对齐 test_tool/send_request_template.py
- 用于直接测试 CAi tool server，不经过 agent
- 推荐先用小样本 smoke test（如 3 个 decoration）跑通链路
"""

import json
import time

import requests

# ============================================================
# 配置区（按需修改）
# ============================================================
WORKER_IP = "127.0.0.1"  # 工具后端服务器 IP
PORT = 8001

TOOL_NAME = "libinvent"  # 对应 tools/<TOOL_NAME>/ 目录名
ACTION = "default"  # config.json 中定义的 action 名；单脚本工具填 "default"

# 发送给工具脚本的参数（会被写入 params.json）
PAYLOAD = {
    "smiles": "CC1(C)S[C@@H]2(NC(=O)[*])C(=O)N2[C@H]1C(=O)O",
    "number_of_decorations_per_scaffold": 3,
    "exclude_smiles": [],
    "batch_size": 1,
    "randomize": True,
    "run_type": "scaffold_decorating",
    "max_rounds": 5,
    "oversample_factor": 3,
    "max_candidates_per_round": 128,
    "preview_limit": 10,
    "include_debug_paths": True,
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
    print(f"[1/4] 提交任务到 {RUN_URL}")
    print("      Payload:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

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
    print(f"[2/4] 等待任务完成（最长 {TIMEOUT_SECS}s）…")
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
            time.sleep(POLL_INTERVAL)


def print_result(result: dict):
    """格式化打印结果。"""
    print("\n[3/4] 结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not isinstance(result, dict):
        print("\n  ❌ 返回结果不是 dict")
        return

    if not result.get("success", False):
        print(f"\n  ❌ 工具内部错误: {result.get('error')}")
        if result.get("error_type"):
            print(f"  error_type: {result.get('error_type')}")
        if "recoverable" in result:
            print(f"  recoverable: {result.get('recoverable')}")
        if result.get("repair_hint"):
            print(f"  repair_hint: {result.get('repair_hint')}")
        return

    summary = result.get("summary", {})
    if summary:
        print("\n  --- Summary ---")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    results = result.get("results")
    if isinstance(results, list):
        print(f"\n  --- Results (前 5 条 / 共 {len(results)} 条) ---")
        for item in results[:5]:
            print(f"  {item}")
    elif isinstance(results, dict):
        print("\n  --- Results ---")
        for key, val in results.items():
            if isinstance(val, list):
                print(f"  {key} (前 5 条):")
                for item in val[:5]:
                    print(f"    {item}")
            else:
                print(f"  {key}: {val}")

    errors = result.get("errors")
    if errors:
        print(f"\n  ⚠️ 部分错误: {errors}")

    print("\n  ✅ 测试成功！")


def print_debug_hint():
    print("\n[4/4] 失败时建议排查：")
    print("  1) 查看 app.py 所在终端输出")
    print("  2) 查看 workspace/jobs/<job_id>/stderr.log")
    print("  3) 查看 workspace/jobs/<job_id>/result.json")
    print("  4) 查看 workspace/jobs/<job_id>/error.json（如果有）")


def main():
    print("=" * 60)
    print(f"工具: {TOOL_NAME} / action: {ACTION}")
    print(f"后端: {BASE_URL}")
    print("=" * 60)

    try:
        print("[0/4] 检查 /tools")
        r = requests.get(f"{BASE_URL}/tools", timeout=10, proxies=NO_PROXY)
        r.raise_for_status()
        print(json.dumps(r.json(), ensure_ascii=False, indent=2))

        job_id = submit_job(PAYLOAD)
        result = poll_job(job_id)
        print_result(result)
        print_debug_hint()

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print_debug_hint()


if __name__ == "__main__":
    main()
