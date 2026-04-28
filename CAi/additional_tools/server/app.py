import asyncio
import json
import sys
from contextlib import asynccontextmanager  # 1. 引入上下文管理器
from pathlib import Path

import uvicorn
from fastapi import BackgroundTasks, FastAPI

# 加载全局配置
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import TOOL_SERVER_HOST, TOOL_SERVER_PORT

# 确保无论从哪个工作目录启动，都能找到同目录下的模块
_SERVER_DIR = Path(__file__).resolve().parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from job_manager import job_manager
from tool_manager import tool_manager


# 2. 定义周期性清理任务
async def _periodic_cleanup():
    """每 24 小时清理一次过期 Job 目录"""
    while True:
        try:
            await asyncio.sleep(86400)
            cleaned = job_manager.cleanup_old_jobs()
            if cleaned:
                print(f"[cleanup] Removed {cleaned} old job directories.")
        except asyncio.CancelledError:
            # 当应用关闭时，这个任务会被取消
            break


# 3. 定义生命周期处理器 (替代 @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: 应用启动前执行 ---
    # 启动时先做一次清理
    job_manager.cleanup_old_jobs()
    # 开启后台定期清理任务
    cleanup_task = asyncio.create_task(_periodic_cleanup())

    yield  # 这里是应用运行的阶段

    # --- Shutdown: 应用关闭前执行 ---
    # 如果需要，可以在这里显式取消任务
    cleanup_task.cancel()
    print("[server] Shutdown complete.")


# 4. 在初始化 FastAPI 时挂载 lifespan
app = FastAPI(lifespan=lifespan)

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = BASE_DIR / "workspace" / "jobs"

# @app.on_event("startup")  <-- 这部分可以删掉了


@app.get("/tools")
def list_tools():
    return {"tools": {name: list(info["scripts"].keys()) for name, info in tool_manager.tools.items()}}


@app.post("/run/{tool}/{action}")
def run_tool(tool: str, action: str, params: dict, background_tasks: BackgroundTasks):
    if tool not in tool_manager.tools:
        return {"error": f"tool not found: {tool}"}

    if action not in tool_manager.tools[tool]["scripts"]:
        return {"error": f"action '{action}' not found in tool '{tool}'"}

    job_id = job_manager.prepare_job(tool, action, params)
    background_tasks.add_task(job_manager.run_job, job_id, tool, action)

    return {"job_id": job_id}


@app.get("/job/{job_id}")
def job_status(job_id: str):
    job_dir = WORKSPACE / job_id

    if not job_dir.exists():
        return {"status": "not_found"}

    result_file = job_dir / "result.json"
    error_file = job_dir / "error.json"

    if result_file.exists():
        with open(result_file, encoding="utf-8") as f:
            result_data = json.load(f)
        return {"status": "finished", "data": result_data}

    if error_file.exists():
        with open(error_file, encoding="utf-8") as f:
            error_data = json.load(f)
        return {"status": "failed", "data": error_data}

    return {"status": "running"}


if __name__ == "__main__":
    uvicorn.run(app, host=TOOL_SERVER_HOST, port=TOOL_SERVER_PORT)
