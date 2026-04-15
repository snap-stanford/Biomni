# 工具开发模板（test_tool）

本目录是新建 CAi 工具的完整模板，包含后端脚本和调试请求脚本。

---

## 目录文件说明

```
test_tool/
├── config.json               ← 工具元信息（必需）
├── run.py                    ← 后端执行脚本 · 单 action 模板
├── sample.py                 ← 后端执行脚本 · 多 action 示例（生成）
├── score.py                  ← 后端执行脚本 · 多 action 示例（打分）
└── send_request_template.py  ← 本地调试脚本（发送请求 + 打印结果）
```

---

## 快速上手：新建一个工具

### 第 1 步：复制本目录

```bash
cp -r additional_tools/server/tools/test_tool \
       additional_tools/server/tools/<your_tool_name>
```

### 第 2 步：修改 `config.json`

```json
{
  "name": "your_tool_name",
  "conda_env": "your_conda_env",
  "gpu": false
}
```

| 字段 | 说明 |
|---|---|
| `name` | 工具唯一 ID，**必须与目录名一致** |
| `conda_env` | 运行脚本的 conda 环境名 |
| `gpu` | `true` = 框架自动申请 GPU 并注入 `CUDA_VISIBLE_DEVICES` |

**多 action 工具**（一个工具、多个脚本）：

```json
{
  "name": "your_tool_name",
  "conda_env": "your_conda_env",
  "gpu": true,
  "actions": {
    "sample": "sample.py",
    "score":  "score.py"
  }
}
```

缺省 `actions` 时框架自动使用 `{"default": "run.py"}`。

---

### 第 3 步：编写后端脚本（`run.py`）

后端脚本只需遵守两条约定：

1. **从 `params.json` 读取参数**（cwd 就是沙盒目录，直接用相对路径）
2. **将结果写入 `result.json`**

`result.json` 标准结构：

```json
{
  "success": true,
  "summary": {
    "task": "描述",
    "input_molecules": 3,
    "processed_molecules": 3
  },
  "results": [...],
  "errors": null
}
```

> **注意**：调试信息写到 `stderr`（`print(..., file=sys.stderr)`），`stdout` 会被 JobManager 捕获到 `stdout.log`，不会返回给 Agent。

---

### 第 4 步：在 `template_tools.py` 中注册 Agent 工具函数

```python
def my_tool_function(smiles_list: list) -> str:
    """
    [工具描述 - LLM 会读这里决定何时调用]
    ...
    """
    if not smiles_list:
        return json.dumps({"success": False, "error": "smiles_list is required"})

    payload = {"smiles_list": smiles_list}

    # 单 action:  _call_worker_api("your_tool_name", payload)
    # 多 action:  _call_worker_api("your_tool_name", payload, action="score")
    result = _call_worker_api("your_tool_name", payload)

    return json.dumps(result, ensure_ascii=False)
```

---

### 第 5 步：启动后端服务

在发送任何请求之前，必须先在**工具服务器**上启动 FastAPI 后端。

**安装依赖（首次）：**

```bash
pip install fastapi uvicorn
```

**启动服务：**

```bash
cd /path/to/Biomni_molecule/CAi
python additional_tools/server/app.py
```

启动后服务监听在 `http://0.0.0.0:8001`，终端输出类似：

```
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
```

可用接口：

| 接口 | 说明 |
|---|---|
| `GET  /tools` | 列出所有已加载的工具 |
| `POST /run/{tool}/{action}` | 提交工具任务，返回 `job_id` |
| `GET  /job/{job_id}` | 查询任务状态和结果 |

验证服务正常并确认你的工具已被加载：

```bash
curl http://<WORKER_IP>:8001/tools
# 返回示例: {"tools": {"test_tool": ["default"], "scscore": ["default"], ...}}
```

> **注意**：每次新增或修改工具脚本后，需要**重启** `app.py` 才能生效。

---

### 第 6 步：调试

**方法 A：curl 直接测试**

```bash
# 提交任务
curl -X POST http://<WORKER_IP>:8001/run/<tool_name>/default \
     -H "Content-Type: application/json" \
     -d '{"smiles_list": ["c1ccccc1"]}'
# 返回: {"job_id": "xxxxxxxx-..."}

# 查询结果
curl http://<WORKER_IP>:8001/job/<job_id>
# 返回: {"status": "finished", "data": {...}}
```

**方法 B：使用本目录的调试脚本**

编辑 `send_request_template.py` 顶部的配置区：

```python
WORKER_IP = "你的服务器IP"
TOOL_NAME = "your_tool_name"
ACTION    = "default"       # 或 "sample" / "score" 等
PAYLOAD   = {"smiles_list": ["c1ccccc1"]}
```

然后运行：

```bash
python send_request_template.py
```

脚本会自动提交任务 → 轮询状态 → 打印结果，适合在开发阶段快速验证。

---

## 沙盒目录结构（供参考）

每个 Job 都在独立的沙盒目录中执行：

```
additional_tools/server/workspace/jobs/<uuid>/
├── params.json   ← JobManager 写入，脚本读取
├── result.json   ← 脚本写入，JobManager / Agent 读取
├── error.json    ← 异常时写入
├── stdout.log    ← 标准输出（print 到这里）
└── stderr.log    ← 标准错误（print(..., file=sys.stderr) 到这里）
```

---

## 完整 Checklist

```
□ 1. tools/<your_tool>/ 目录下有 config.json 和 run.py（或多 action 脚本）
□ 2. conda 环境已安装所有依赖
□ 3. template_tools.py 中已添加 Agent 工具函数，docstring 清晰
□ 4. pip install fastapi uvicorn（首次）
□ 5. python additional_tools/server/app.py 启动后端服务
□ 6. curl http://<IP>:8001/tools 确认工具已被加载
□ 7. 用 send_request_template.py 或 curl 手动测试一次，验证结果格式
```
