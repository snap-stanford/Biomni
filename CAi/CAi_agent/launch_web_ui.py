"""启动脚本 - 同时启动 FastAPI 后端和 React 前端"""

import subprocess
import sys
import time
from pathlib import Path

# 加载全局配置
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import WEB_BACKEND_HOST, WEB_BACKEND_PORT, WEB_FRONTEND_PORT


def check_dependencies():
    """检查必要的依赖"""
    try:
        import fastapi
        import uvicorn
        import websockets
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请安装: pip install fastapi uvicorn websockets")
        sys.exit(1)

    frontend_dir = Path(__file__).parent / "web_ui" / "frontend"
    if not (frontend_dir / "node_modules").exists():
        print("❌ 前端依赖未安装")
        print(f"请在 {frontend_dir} 目录下运行: npm install")
        sys.exit(1)


def start_backend(agent, host=WEB_BACKEND_HOST, port=WEB_BACKEND_PORT):
    """启动后端 API"""
    from .web_ui.backend.api import create_api

    api = create_api(agent)
    print(f"🚀 Starting backend on http://localhost:{port}")

    import uvicorn

    uvicorn.run(api.app, host=host, port=port)


def start_frontend():
    """启动前端开发服务器"""
    frontend_dir = Path(__file__).parent / "web_ui" / "frontend"

    print(f"🚀 Starting frontend on http://localhost:{WEB_FRONTEND_PORT}")
    print(f"📁 Frontend directory: {frontend_dir}")

    subprocess.run(["npm", "run", "dev"], cwd=frontend_dir)


def launch_web_ui(agent, backend_port=WEB_BACKEND_PORT):
    """
    启动完整的 Web UI（后端 + 前端）

    Args:
        agent: A1pro agent 实例
        backend_port: 后端 API 端口（默认读取 config.py / .env）
    """
    import threading

    print("\n" + "=" * 60)
    print("🌐 CAi Agent Web UI")
    print("=" * 60)

    check_dependencies()

    # 在单独的线程中启动后端
    backend_thread = threading.Thread(target=start_backend, args=(agent,), kwargs={"port": backend_port}, daemon=True)
    backend_thread.start()

    # 等待后端启动
    time.sleep(2)

    print("\n✅ Backend started successfully")
    print(f"📡 API: http://localhost:{backend_port}")
    print(f"📡 WebSocket: ws://localhost:{backend_port}/ws/chat")
    print("\n🎨 Starting frontend...\n")

    # 启动前端（阻塞主线程）
    try:
        start_frontend()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        sys.exit(0)


if __name__ == "__main__":
    print("请使用 agent.launch_web_ui() 来启动 Web UI")
