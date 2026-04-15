
---

# 🛠️ 外部工具接入指南 (Tool Integration Guide)

本文档旨在指导开发者如何将新的科学计算脚本、AI 推理模型或外部程序无缝集成到我们的工具调用调度系统中。

## 📐 核心设计理念：沙盒隔离与文件 I/O

为了支持高并发并防止多个任务互相干扰，本系统采用了**“按任务隔离的沙盒目录”**设计：

1. 每次外部请求调用工具时，系统都会生成一个全局唯一的任务 ID（UUID）。
2. 系统会创建一个专属的空文件夹：`workspace/jobs/<job_id>/`。
3. 系统会将 API 接收到的参数保存为该目录下的 `params.json`。
4. **关键点：** 系统会将你的脚本的**当前工作目录 (CWD)** 切换到这个专属文件夹中运行。
5. 你的脚本执行完毕后，必须将结构化结果写入当前目录下的 `result.json` 中。

---

## 📂 目录结构规范

要添加一个新工具，请在 `tools/` 目录下创建一个以你的**工具名称**命名的新文件夹。该文件夹内必须包含两个核心文件：`config.json` 和 `run.py`。

示例结构：

```text
tools/
└── your_new_tool/         <-- 你的工具名称 (即 API 路由中的 {tool})
    ├── config.json        <-- 环境与资源配置文件
    ├── run.py             <-- 核心执行入口脚本
    └── 源文件...      <-- (也就是代码本体)

```

---

## 📝 步骤 1：编写 `config.json`

这个文件告诉调度系统应该用什么环境、分配什么硬件资源来运行你的脚本。

```json
{
  "conda_env": "my_env_name",
  "gpu": true
}

```

* **`conda_env`** (必填): 运行该脚本所需的 conda 虚拟环境名称。系统会自动使用 `conda run -n <env_name>` 来启动脚本。
* **`gpu`** (可选): 布尔值。如果设置为 `true`，调度系统会自动为你分配一块空闲的 GPU，并注入 `CUDA_VISIBLE_DEVICES` 环境变量。如果设置为 `false` 或不写，则不分配 GPU。

---

## 🚀 步骤 2：编写 `run.py`

这是工具的主入口文件。你的脚本必须遵循以下**输入输出规范**：

### 📥 1. 如何读取参数？

参数固定保存在当前目录的 `params.json` 中。**请直接使用相对路径读取**，不要使用绝对路径去工具源码目录找。

### 📤 2. 如何返回结果？

无论任务成功还是因为业务逻辑报错（如输入的 SMILES 不合法），都**必须**将结果以 JSON 格式写入当前目录的 `result.json` 中。

* **绝对不要**依赖 `print()` 或 `sys.stdout` 来传递需要被解析的结构化数据（底层 C++ 库的 Warning 会破坏 JSON 格式）。
* `print()` 仅用于打印调试日志，这些内容会被系统收集到 `stdout.log` 和 `stderr.log` 中以备查阅。

### 💻 `run.py` 标准代码模板：

```python
import json
from pathlib import Path

#todo
def process_data(params):
    """
    你的核心业务逻辑写在这里,也就是对外暴露的工具
    """
    # 示例：获取参数
    input_data = params.get("input_data")
    if not input_data:
        raise ValueError("缺少必要参数: input_data")

    # 执行计算...
    score = 99.9

    # 也可以在当前沙盒目录下生成其他文件（互不干扰）
    with open("temp_output.csv", "w") as f:
        f.write("id,score\n1,99.9\n")

    return {
        "score": score,
        "csv_file": "temp_output.csv"
    }

def main():
    # 最终要写入 result.json 的字典
    result_payload = {}

    try:
        # 1. 强制从当前沙盒目录读取 params.json
        params_file = Path("params.json")
        if not params_file.exists():
            raise FileNotFoundError("当前沙盒目录下未找到 params.json")

        with open(params_file, "r", encoding="utf-8") as f:
            params = json.load(f)

        # 2. 运行核心逻辑
        data = process_data(params)

        # 3. 构造成功响应
        result_payload = {
            "success": True,
            "data": data
        }

    except Exception as e:
        # 4. 捕获一切内部错误，构造失败响应
        result_payload = {
            "success": False,
            "error": str(e)
        }

    # 5. 将结果写入当前沙盒目录的 result.json
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    # 这些内容会进入 stdout.log
    if result_payload.get("success"):
        print("🎉 工具运行成功，结果已保存至 result.json")
    else:
        print(f"❌ 工具运行失败: {result_payload.get('error')}")

if __name__ == "__main__":
    main()

```

---

## ⚠️ 开发者避坑指南 (Checklist)

1. **🚫 禁用绝对路径硬编码用于 I/O：** 输出的中间文件、日志、图表等，一律直接写在**当前工作目录 (`./`)** 下。调度器已经帮你隔离好了房间，写在当前目录就是最安全的。
2. **✅ 模型权重的加载：** 如果你的工具需要加载巨大的预训练模型权重（`.pt`, `.ckpt` 等），**请使用绝对路径或基于工具源码所在路径的相对路径**来定位模型文件（只读操作没有并发冲突风险）。
3. **✅ `result.json` 的 `"success"` 字段：** 最外层务必保留 `"success": true/false` 的布尔标志，方便前端或 Main Agent 直接判断任务流转状态。

---
