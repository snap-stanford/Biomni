# Biomni Molecule - CAi

CAi 是基于 Biomni A1 框架的增强版药物分子 AI Agent，集成了分子生成、对接、毒性预测等多种计算化学工具，支持通过自然语言驱动复杂的药物发现工作流。

---

## 项目结构

```
Biomni_molecule/
├── CAi/
│   ├── config.py                        # 全局配置（端口、LLM 参数）
│   ├── .env                             # 本地环境变量（填写 API Key，不提交 git）
│   ├── main.py                          # Agent 启动入口
│   ├── additional_tools/
│   │   ├── __init__.py
│   │   ├── template_tools.py            # Agent 可调用的工具函数
│   │   └── server/
│   │       ├── app.py                   # 工具执行后端（FastAPI）
│   │       ├── job_manager.py           # Job 沙盒管理
│   │       ├── install_all.sh           # 一键安装所有工具 conda 环境
│   │       └── tools/                  # 各工具目录（config.json + run.py）
│   └── CAi_agent/
│       ├── agent.py                     # A1pro Agent 类
│       ├── ui.py                        # Gradio UI
│       └── skills/                      # Agent 技能文件
└── base_CAi/                            # 基础 Agent 框架
```

---

## 快速开始

### 第一步：配置环境变量

在 `CAi/` 目录下创建 `.env` 文件：

```bash
# CAi/.env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=http://35.220.164.252:3888/v1/
LLM_MODEL=claude-sonnet-4-5-20250929

TOOL_SERVER_HOST=0.0.0.0
TOOL_SERVER_PORT=8001
```

### 第二步：安装基础依赖

```bash
pip install fastapi uvicorn python-dotenv
```

### 第三步：安装工具 conda 环境

每个计算工具运行在独立的 conda 环境中。可以按需安装：

```bash
cd CAi/additional_tools/server

# 安装全部工具环境（较慢）
bash install_all.sh

# 只安装部分工具
bash install_all.sh vina scscore toxicity
```

### 第四步：启动工具后端服务

```bash
# 在 CAi/ 目录下运行
python additional_tools/server/app.py
```

服务启动后监听 `http://0.0.0.0:8001`，可用接口：
- `GET  /tools`               — 列出所有已加载工具
- `POST /run/{tool}/{action}` — 提交工具任务
- `GET  /job/{job_id}`        — 查询任务状态

### 第五步：启动 Agent UI

```bash
# 在 Biomni_molecule/ 目录下运行
python CAi/main.py
```

或自定义模型：

```python
# CAi/main.py 中修改 llm 参数
agent = A1pro(
    llm="Qwen/Qwen3-32B",       # 替换为你的模型
    source="Custom",
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    auto_load_tools=True,
)
agent.launch_new_gradio_demo(share=False)
```

---

## 可用工具

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

## 添加新工具

每个工具需要三个步骤：

**1. 创建工具目录**

```
additional_tools/server/tools/<your_tool>/
├── config.json    # 声明 conda 环境、GPU 需求
└── run.py         # 读 params.json → 计算 → 写 result.json
```

`config.json` 示例：

```json
{
  "name": "mytool",
  "conda_env": "mytool_env",
  "gpu": false
}
```

**2. 编写 `run.py`**

```python
import json

def main():
    params = json.load(open("params.json"))
    # ... 计算逻辑 ...
    result = {"success": True, "summary": {...}, "results": [...]}
    with open("result.json", "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
```

**3. 在 `template_tools.py` 中注册 Agent 工具函数**

```python
def my_tool(smiles: str) -> str:
    """工具描述（大模型根据此决定何时调用）"""
    payload = {"smiles": smiles}
    result = _call_worker_api("mytool", payload)
    return json.dumps(result, ensure_ascii=False)
```

详细开发指南见 [CAi/start.md](CAi/start.md)。

---

## 架构说明

工具调用链路：

```
Agent (template_tools.py)
    │  POST /run/{tool}/{action}
    ▼
FastAPI (app.py)  →  JobManager
    │  conda run -n <env> python run.py
    │  cwd = workspace/jobs/<uuid>/
    ▼
run.py  →  result.json
    ▼
Agent 收到结果
```

每个 Job 运行在独立沙盒目录，互不干扰，支持 GPU 自动分配。
