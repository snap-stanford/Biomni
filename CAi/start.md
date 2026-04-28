# CAi 启动指南

## 项目结构

```
Biomni_molecule/
├── CAi/
│   ├── config.py                        # 全局配置（端口、LLM 参数）
│   ├── .env                             # 本地环境变量（填写 API Key）
│   ├── additional_tools/
│   │   ├── __init__.py                  # 工具导出
│   │   ├── template_tools.py            # Agent 可调用的工具函数（含服务器地址）
│   │   └── server/
│   │       └── app.py                   # 工具执行后端服务（FastAPI）
│   └── CAi_agent/
│       ├── agent.py                     # A1pro Agent 类
│       ├── ui.py                        # Gradio UI
│       └── launch_web_ui.py             # Web UI 启动脚本
└── base_CAi/                            # 基础 Agent 框架
```

---

## 第一步：配置环境变量

在 `CAi/` 根目录下创建 `.env` 文件，填写 API Key 和服务地址：

```bash
# CAi/.env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=http://35.220.164.252:3888/v1/   # 或你自己的 LLM 服务地址
LLM_MODEL=claude-sonnet-4-5-20250929

TOOL_SERVER_HOST=0.0.0.0
TOOL_SERVER_PORT=8001
```

---

## 第二步：安装依赖

```bash
pip install fastapi uvicorn
```

---

## 第三步：启动工具后端服务

工具后端负责接收 Agent 的工具调用请求（分子生成、对接、毒性预测等）并异步执行。

```bash
cd /path/to/Biomni_molecule/CAi
python additional_tools/server/app.py
```

启动后服务监听在 `http://0.0.0.0:8001`，可用接口：
- `GET  /tools`              — 列出所有可用工具
- `POST /run/{tool}/{action}` — 提交工具任务
- `GET  /job/{job_id}`       — 查询任务状态

---

## 第四步：配置 Agent 工具服务器地址

`additional_tools/template_tools.py` 第 7-9 行控制 Agent 实际连接的工具服务器地址：

```python
TOOL_SERVER_HOST = os.environ.get("TOOL_SERVER_HOST", "100.103.118.42")
TOOL_SERVER_PORT = os.environ.get("TOOL_SERVER_PORT", "8001")
BASE_URL = f"http://{TOOL_SERVER_HOST}:{TOOL_SERVER_PORT}"
```

**修改方式（推荐）**：在 `.env` 中设置 `TOOL_SERVER_HOST` 为运行后端服务的机器 IP。

---

## 第五步：启动 Agent（Gradio UI）

```python
# run_agent.py（在 Biomni_molecule/ 目录下运行）
import os
from CAi.CAi_agent.agent import A1pro
from CAi.config import LLM_BASE_URL,LLM_API_KEY

agent = A1pro(
    # llm="claude-sonnet-4-5-20250929",
    # llm = "Qwen/Qwen3-8B",
    llm = "Qwen/Qwen3-32B",
    # llm = "deepseek-ai/DeepSeek-R1",
    # llm = "x-ai/grok-4-fast",
    source="Custom",
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    use_tool_retriever=False,
    expected_data_lake_files=[],
    auto_load_tools=True,  # 自动加载工具
)

agent.launch_new_gradio_demo(share=True)
```


---

## 快速启动（汇总）

```bash
# 终端 1：启动工具后端
cd /path/to/Biomni_molecule/CAi
python additional_tools/server/app.py

# 终端 2：启动 Agent UI
cd projects/Biomni_molecule/CAi/
python test_web.py
```

---

## 可用工具列表

| 工具函数 | 功能 |
|---|---|
| `generate_scaffold_analogs` | 骨架衍生生成 |
| `predict_molecule_toxicity` | 毒性预测 + SHAP 解释 |
| `calculate_scscore` | 合成可行性评分（SCScore） |
| `generate_libinvent_decorations` | Lib-INVENT 骨架修饰 |
| `predict_antibacterial_pmic` | 抗菌活性（pMIC）预测 |
| `generate_molecules_for_pocket` | RxnFlow 靶点口袋导向生成 |
| `perform_molecular_docking_vina` | AutoDock Vina 分子对接 |
| `score_molecules_reinvent` | REINVENT4 多维综合打分 |
| `generate_molecules_reinvent` | REINVENT4 从头分子生成 |
| `generate_molecules_drugex` | DrugEx 图网络强化学习生成 |

---

# 后端工具开发指南

> 以 `scscore` 为例，完整介绍如何为 CAi 编写一个新的后端工具。

## 整体架构：沙盒运行机制

Agent 调用工具的完整链路如下：

```
Agent (template_tools.py)
    │  POST /run/{tool}/{action}  params: {...}
    ▼
FastAPI (app.py)
    │  创建 Job 目录，写入 params.json
    │  后台任务: job_manager.run_job(job_id, tool, action)
    ▼
JobManager (job_manager.py)
    │  conda run -n <env> python <script.py>
    │  cwd = workspace/jobs/<job_id>/         ← 隔离沙盒目录
    │  stdin = params.json 内容 (JSON string)
    ▼
工具脚本 (run.py)
    │  从 params.json 读取参数
    │  执行计算逻辑
    │  将结果写入 result.json
    ▼
JobManager 轮询 result.json / error.json
    ▼
Agent 收到结果，返回给大模型
```

**关键设计：每个 Job 都有独立的沙盒目录**

```
additional_tools/server/workspace/jobs/
└── <uuid>/
    ├── params.json     ← 输入参数（由 JobManager 写入）
    ├── result.json     ← 计算结果（由 run.py 写入）
    ├── error.json      ← 错误信息（由 run.py 或 JobManager 写入）
    ├── stdout.log      ← 标准输出（由 JobManager 捕获）
    └── stderr.log      ← 标准错误（由 JobManager 捕获）
```

脚本的工作目录（`cwd`）就是这个 uuid 目录，所以直接 `open("params.json")` 和 `open("result.json", "w")` 即可，**不需要写绝对路径**。

---

## 文件结构：新建一个工具需要的三个文件

```
additional_tools/server/tools/
└── <your_tool_name>/           ← 工具目录，名字即工具 ID
    ├── config.json             ← 必需：声明环境、GPU 需求、action 映射
    ├── run.py                  ← 主脚本（单 action 工具）
    └── <依赖代码或模型>/
```

---

## 第一步：编写 `config.json`

### 最简单的情况（单 action，不需要 GPU）

```json
{
  "name": "scscore",
  "conda_env": "scscore",
  "gpu": false
}
```

- `name`：工具名，与目录名保持一致
- `conda_env`：运行脚本所用的 conda 环境名
- `gpu`：是否需要向 GPU 队列申请显卡（`false` = CPU 工具）

当 `actions` 字段缺省时，框架自动使用 `{"default": "run.py"}`，即 `POST /run/scscore/default` 会执行 `run.py`。

### 需要 GPU 的情况

```json
{
  "name": "mytool",
  "conda_env": "mytool_env",
  "gpu": true
}
```

`gpu_manager` 会从可用 GPU 列表中取一张卡，通过 `CUDA_VISIBLE_DEVICES` 注入环境变量，任务结束后自动归还。

### 多 action 的情况（一个工具、多个脚本）

```json
{
  "name": "reinvent4",
  "conda_env": "reinvent4",
  "gpu": true,
  "actions": {
    "sample": "sample.py",
    "score":  "score.py"
  }
}
```

这样 Agent 可以分别调用 `POST /run/reinvent4/sample` 和 `POST /run/reinvent4/score`。

---

## 第二步：编写 `run.py`（以 scscore 为完整示例）

`run.py` 只需遵守一个约定：**从 `params.json` 读参数，把结果写到 `result.json`**。

```python
import sys
import json
from pathlib import Path

def main():
    # ✅ 1. 读取参数（固定写法）
    # 脚本的 cwd 就是 Job 沙盒目录，直接读 params.json
    params = json.load(open("params.json"))

    smiles_list = params.get("smiles_list", [])
    model_type  = params.get("model_type", "1024bool")

    # ✅ 2. 执行计算逻辑
    # ... 你的核心代码 ...
    result = calculate_scscore(smiles_list, model_type)

    # ✅ 3. 写入结果（固定写法）
    # result 必须是一个可 JSON 序列化的 dict
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
```

### result.json 的格式约定

Agent 端（`template_tools.py`）会解析 `result.json`，建议遵守以下结构：

```json
{
  "success": true,
  "summary": {
    "total": 2,
    "successful": 2,
    "failed": 0,
    "avg_scscore": 1.85
  },
  "results": [...],
  "errors": null
}
```

| 字段 | 说明 |
|---|---|
| `success` | 布尔值，`false` 时 Agent 会报错 |
| `summary` | 统计摘要，Agent 优先读这里（避免传输大量数据） |
| `results` | 完整的逐条结果列表 |
| `errors` | 部分失败的错误列表，全部成功时为 `null` |

### 错误处理：用 try/except 包裹 main()

```python
def main():
    try:
        params = json.load(open("params.json"))
        # ... 正常逻辑 ...
        result = {"success": True, "summary": {...}, "results": [...]}
    except Exception as e:
        # 任何异常都写入 result.json，JobManager 会识别 success=False
        result = {"success": False, "error": str(e)}

    with open("result.json", "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
```

> **注意**：不要用 `print()` 输出结果，stdout 会被 `JobManager` 捕获到 `stdout.log`，并不会被 Agent 读取。调试信息用 `print(..., file=sys.stderr)` 写到 stderr。

---

## 第三步：在 `template_tools.py` 中注册 Agent 工具函数

后端脚本写好之后，需要在 `additional_tools/template_tools.py` 里增加一个 Python 函数，Agent 才能调用。

```python
def calculate_scscore(smiles: str = None, smiles_list: list = None, model_type: str = "1024bool") -> str:
    """
    [工具描述 - 大模型会读这里来决定何时调用]
    Estimate the synthetic accessibility of molecules using the SCScore model.
    ...
    """
    # 1. 参数校验
    if smiles:
        smiles_list = [smiles]
    if not smiles_list:
        return json.dumps({"success": False, "error": "smiles or smiles_list must be provided"})

    # 2. 调用后端
    payload = {"smiles_list": smiles_list, "model_type": model_type}
    result = _call_worker_api("scscore", payload)   # 工具名对应 tools/ 目录名
    #                                    ↑ 如果有多 action: _call_worker_api("reinvent4", payload, action="score")

    # 3. 格式化返回给 Agent 的字符串
    return json.dumps(result, ensure_ascii=False)
```

`_call_worker_api` 内部会自动轮询 Job 状态，直到 `result.json` 出现，默认超时 5 分钟，可用 `timeout_mins` 参数调整。

---

## 完整新建工具 Checklist

```
□ 1. 在 tools/<your_tool>/ 目录下创建以下文件：
      - config.json  （填写 name / conda_env / gpu）
      - run.py       （读 params.json → 计算 → 写 result.json）

□ 2. 确保对应 conda 环境已安装依赖
      conda activate <env>
      pip install ...

□ 3. 在 template_tools.py 中添加 Agent 工具函数
      - 函数名会被 Agent 识别为工具名
      - docstring 是 Agent 理解如何使用该工具的依据，务必清晰

□ 4. 重启后端服务后，访问 GET /tools 确认工具已被加载
      curl http://localhost:8001/tools

□ 5. 用 curl 手动测试一次
      curl -X POST http://localhost:8001/run/scscore/default \
           -H "Content-Type: application/json" \
           -d '{"smiles_list": ["c1ccccc1"]}'
      # 返回 {"job_id": "..."}

      curl http://localhost:8001/job/<job_id>
      # 返回 {"status": "finished", "data": {...}}
```
