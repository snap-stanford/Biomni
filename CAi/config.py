"""
全局配置文件
所有端口、主机地址、LLM 参数统一在此管理。
修改方式：
  1. 直接修改本文件中的默认值（适合开发环境）
  2. 在 biomiplus/.env 中设置对应环境变量（推荐，优先级更高）
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件
load_dotenv(Path(__file__).parent / ".env")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent


# =============================================================================
# 端口配置
# =============================================================================

# additional_tools/server/app.py  —— 工具执行服务
TOOL_SERVER_HOST = os.getenv("TOOL_SERVER_HOST", "0.0.0.0")
TOOL_SERVER_PORT = int(os.getenv("TOOL_SERVER_PORT", 8001))

# bio_agent/web_ui/backend/api.py  —— Web UI 后端（Vite proxy 指向此端口）
WEB_BACKEND_HOST = os.getenv("WEB_BACKEND_HOST", "0.0.0.0")
WEB_BACKEND_PORT = int(os.getenv("WEB_BACKEND_PORT", 8000))

# bio_agent/web_ui/frontend (Vite dev server)
WEB_FRONTEND_PORT = int(os.getenv("WEB_FRONTEND_PORT", 3000))

# =============================================================================
# LLM 配置
# =============================================================================

LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-5-20250929")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://35.220.164.252:3888/v1/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # 请在 .env 中填写，勿提交至 git
LLM_SOURCE = os.getenv("LLM_SOURCE", "Custom")
