# CAi 分子设计智能助手 CAi Molecule Design Copilot

这是一个集分子生成、性质评估与候选筛选于一体的智能 Agent 平台，支持基于骨架的分子生成、从头分子设计、靶点导向评估、毒性预测、抗菌活性预测和可合成性分析，实现“生成—评估—筛选”的一体化流程，显著降低了复杂工作流的使用门槛。它是专门为药物发现研究人员和计算化学工作者设计，支持用户以简单命令部署并运行复杂的分子设计工作流，从而免去环境配置、模型集成以及评估流程搭建所带来的负担。这个智能助手面向药物发现与计算化学研究中的实际需求，解决了传统流程中部署复杂、工具分散、多模型衔接困难以及结果难复现等问题。


## 核心亮点

1. 简单快速部署，实现端到端分子设计
2. 面向化学研究者的 Web 交互界面
3. 集成式全流程分子生成、评估与筛选
4. 灵活的工具选择与流程化组合能力

## 工具说明
| 功能类型     | 工具                                        | 函数                      | 详细说明                                                                                             |
| -------- | ----------------------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 骨架约束分子生成 | RNN-based Constrained Scaffold Generation | `generate_scaffold_analogs` | 输入骨架结构，修改部分可以包括R-groups或者linking，在保留核心骨架的前提下生成结构类似的小分子。适用于先导化合物扩展、母核优化和定向类似物探索。          
|
| 骨架约束分子生成 | LibINVENT                                 | `generate_libinvent_decorations`            | 以骨架为中心生成可修饰的分子库，尤其适合围绕固定母核开展系统化 R-group 扩展和可控分子设计，并支持反应类型约束以提升分子合成可行性。                                   
|
| 骨架约束分子生成 | Reinvent 4                                | `generate_molecules_reinvent4_libinvent`    | 更进一步，在多目标打分函数引导下，基于骨架生成具备反应约束的化学分子库。
|
| 从头分子设计   | RXNFlow                                   | `run_rxnflow_design()`                  | 不依赖固定骨架，基于目标蛋白、靶点信息或指定化学空间进行从头设计的小分子生成，适用于靶点导向药物设计与全新候选分子发现。                                   |
| 从头分子设计   | Reinvent 4                                | `generate_molecules_reinvent4_denovo` | 在多目标打分函数引导下，实现多目标驱动的分子生成。                        
|
| 从头分子生成   | Reinvent 4                                | `generate_molecules_reinvent4_mol2mol` |接收完整分子输入并以该分子为条件，在多目标优化驱动下生成结构相似的候选分子，实现局部优化。                         |
| 逆向合成评估   | SC Score                                  | `calculate_scscore`                  | 对生成分子的结构可合成性进行评估，衡量其与已知合成模式的一致性和潜在可行性，可用于候选分子的初步可合成性筛选。                                            |
| 亲和力性能评估  | Vina Score                                | `perform_molecular_docking_vina`                | 需要蛋白质和小分子文件作为输入，计算 docking score，用于预测分子与目标蛋白之间的结合亲和力，支持靶点导向候选分子的筛选与排序。                                |
| ADMET性能评估  | Toxicity Prediction                       | `predict_molecule_toxicity`                    | 使用Toxcast肝细胞毒性数据对模型微调后，用于预测分子的肝细胞毒性反应风险，为早期药物筛选提供安全性参考，并且支持子结构的Shapley value 可视化分析，展示不同子结构对毒性的贡献程度。                                       
| 抑菌浓度性能评估  | MIC Prediction                            | `predict_antibacterial_pmic`                         | 基于化学性质预测模型使用ChEMBL中的所有包括MIC数据的分子训练，并预测分子的最低抑菌浓度（MIC），并辅助抗菌药物设计任务中的分子排序与筛选。                                 |

## 输入示范

### 基于骨架的分子生成
```text
给定青霉素母核骨架
CC1(C)S[C@@H]2(NC(=O)*)C(=O)N2[C@H]1C(=O)O，
使用LibINVENT 和  RNN-based Constrained Scaffold Generation 分别生成 10 个基于骨架的小分子类似物，并按照 SC score进行排序。
```
### 基于靶点的从头生成
```text
以 HIV-1 protease 作为目标蛋白，使用 1HVR.pdb 作为目标结构文件，结合位点中心坐标为 [15.2,23.5,6.8]，调用 Rxnflow 和 Reinvent4 工具生成候选小分子，并按照 Vina score排名。
```
### 性能合成评估
```text
对前面生成的小分子计算 toxicity、MIC，用于评估分子的化学性质成药过程中可能表现出的性质。
```
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
LLM_BASE_URL=your_llm_base_url_here
LLM_MODEL=claude-sonnet-4-5-20250929

TOOL_SERVER_HOST=0.0.0.0
TOOL_SERVER_PORT=8001
```

### 第二步：安装基础依赖

```bash
conda create -n CAi python==3.11
conda activate CAi 
pip install -e .
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
