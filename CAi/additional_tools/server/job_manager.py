"""job_manager.py"""

import asyncio
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

# 确保无论从哪个工作目录启动，都能找到同目录下的模块
_SERVER_DIR = Path(__file__).resolve().parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from gpu_manager import gpu_manager
from tool_manager import tool_manager

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = BASE_DIR / "workspace" / "jobs"
WORKSPACE.mkdir(parents=True, exist_ok=True)


class JobManager:
    # ⭐ 修改 1：增加 action 参数
    def prepare_job(self, tool_name, action, params):
        """只负责创建文件夹和写入参数，不直接运行。这个由于是纯文件操作，保持同步即可"""
        job_id = str(uuid.uuid4())
        job_dir = WORKSPACE / job_id
        job_dir.mkdir()

        # （可选）把 action 也存进 params 里，方便排错或工具内部读取
        params["_action"] = action

        params_file = job_dir / "params.json"
        with open(params_file, "w") as f:
            json.dump(params, f)

        return job_id

    # ⭐ 修改 2：增加 action 参数
    async def run_job(self, job_id, tool_name, action):
        """实际运行逻辑，由 FastAPI 后台调用"""
        tool = tool_manager.get(tool_name)
        job_dir = WORKSPACE / job_id

        with open(job_dir / "params.json") as f:
            params = json.load(f)

        env = os.environ.copy()
        gpu = None
        process = None

        try:
            # 异步等待 GPU，释放线程控制权
            if tool["config"].get("gpu"):
                while True:
                    gpu = gpu_manager.acquire()
                    if gpu is not None:
                        break
                    # 让出事件循环，不阻塞主线程
                    await asyncio.sleep(2)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)

            # ⭐ 修改 3：根据 action 动态获取要执行的脚本路径
            if action not in tool["scripts"]:
                raise ValueError(f"Action '{action}' is not configured for tool '{tool_name}'")

            script_path = tool["scripts"][action]

            cmd = [
                "conda",
                "run",
                "-n",
                tool["env"],
                "python",
                script_path,  # 使用动态获取的脚本路径
            ]

            # 使用异步子进程
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=job_dir,
            )

            # 结合 asyncio.wait_for 实现强制超时控制
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(input=json.dumps(params).encode("utf-8")), timeout=3600
                )
                stdout = stdout_bytes.decode("utf-8")
                stderr = stderr_bytes.decode("utf-8")
                returncode = process.returncode

            except TimeoutError:
                # 如果超时，强杀子进程
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                stdout = ""
                stderr = "Process forcefully killed due to timeout (3600s)."
                returncode = -1

            # 记录日志
            with open(job_dir / "stdout.log", "w", encoding="utf-8") as f:
                f.write(stdout)
            with open(job_dir / "stderr.log", "w", encoding="utf-8") as f:
                f.write(stderr)

            # 兜底机制：非正常退出且没有结果文件
            if returncode != 0 and not (job_dir / "result.json").exists():
                with open(job_dir / "error.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {"success": False, "error": f"Process failed with return code {returncode}", "stderr": stderr},
                        f,
                    )

        except Exception as e:
            # 捕获调度器自身的异常 (如找不到 conda 环境、字典报错等)
            with open(job_dir / "error.json", "w", encoding="utf-8") as f:
                json.dump({"success": False, "error": f"Job manager crashed: {str(e)}"}, f)

        finally:
            # ⭐ 无论子进程是正常结束、崩溃还是超时被杀，100% 保证归还显卡
            if gpu is not None:
                gpu_manager.release(gpu)

    def cleanup_old_jobs(self, max_age_days: int = None) -> int:
        """
        清理超过指定天数的旧 Job 目录。
        max_age_days 默认读取环境变量 JOB_MAX_AGE_DAYS，若未设置则为 7 天。
        返回已清理的目录数量。
        """
        if max_age_days is None:
            max_age_days = int(os.environ.get("JOB_MAX_AGE_DAYS", 7))
        cutoff = time.time() - max_age_days * 86400
        cleaned = 0
        for job_dir in WORKSPACE.iterdir():
            if not job_dir.is_dir():
                continue
            # 仅清理已完成（存在 result.json 或 error.json）的 Job
            is_done = (job_dir / "result.json").exists() or (job_dir / "error.json").exists()
            if is_done and job_dir.stat().st_mtime < cutoff:
                try:
                    shutil.rmtree(job_dir)
                    cleaned += 1
                except Exception as e:
                    print(f"Warning: failed to remove job dir {job_dir}: {e}")
        return cleaned


job_manager = JobManager()
