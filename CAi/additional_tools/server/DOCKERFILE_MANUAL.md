# Tool Server — Dockerfile 操作手册

## 目录结构约定

```
CAi/                                  ← Docker build context（必须从这里执行构建）
├── config.py                         ← 服务器配置（端口等）
└── additional_tools/server/
    ├── Dockerfile                    ← 本文件对应的构建文件
    ├── docker-compose.yml
    ├── install_all.sh                ← 宿主机一键安装所有工具 conda 环境
    ├── app.py                        ← FastAPI 主进程（运行在 scscore 环境中）
    ├── job_manager.py
    ├── tool_manager.py
    ├── gpu_manager.py
    └── tools/
        ├── <tool_name>/
        │   ├── config.json           ← 工具声明（环境名、是否用 GPU）
        │   ├── run.py                ← 工具入口脚本
        │   ├── install.sh            ← 宿主机单独安装该工具 conda 环境的脚本
        │   ├── environment.yml       ← conda 环境依赖（推荐）
        │   └── <lib_dir>/            ← 工具依赖的本地库或模型文件
        └── ...
```

---

## 核心概念：工具如何与 Dockerfile 挂钩

每个工具目录下的 `config.json` 决定运行时行为：

```json
{
  "name": "my_tool",
  "conda_env": "my_env",    ← 必须与 Dockerfile 中创建的环境名完全一致
  "gpu": true               ← true = 运行时自动分配 CUDA_VISIBLE_DEVICES
}
```

`job_manager.py` 实际执行的命令是：
```bash
conda run -n <conda_env> python <script_path>
```

**因此：Dockerfile 中的环境名 = `install.sh` 中的环境名 = `config.json` 中的 `conda_env` 字段。**

---

## Dockerfile 的缓存分层结构

当前 Dockerfile 采用**两阶段 COPY** 策略，以最大化 Docker 层缓存命中率：

```
阶段 1：COPY environment.yml → 创建 conda 环境（每个工具独立 RUN 块）
         ↑ 只要 yml 不变，无论代码改多少次，这些层都命中缓存
阶段 2：COPY 全部代码（run.py、模型文件、config.json 等）
         ↑ 频繁变更，但层轻量，重新 COPY 很快
```

实际对应 Dockerfile 中的结构：

```dockerfile
# 第 4 节：只 COPY yml 文件到 /tmp/envs/
COPY additional_tools/server/tools/my_tool/environment.yml /tmp/envs/my_tool.yml

# 第 5 节：从 yml 创建环境（重量级，缓存）
RUN conda env create -n my_env -f /tmp/envs/my_tool.yml && conda clean -afy

# 第 6 节：COPY 源码（在所有环境安装完成后）
COPY additional_tools/server/tools /app/server/tools
```

---

## 添加新工具的完整步骤

### 第一步：创建工具目录

```
tools/
└── my_new_tool/
    ├── config.json
    ├── run.py
    ├── install.sh            ← 宿主机安装脚本（参考现有工具）
    ├── environment.yml       ← conda 环境导出文件（推荐）
    └── my_lib/               ← 如果有本地依赖库或模型文件
```

`config.json` 最小示例：
```json
{
  "name": "my_new_tool",
  "conda_env": "my_new_env",
  "gpu": false
}
```

如果工具有多个动作（action），参考 reinvent4 的扩展写法：
```json
{
  "name": "my_new_tool",
  "conda_env": "my_new_env",
  "gpu": false,
  "actions": {
    "predict": "predict.py",
    "evaluate": "evaluate.py"
  }
}
```

---

### 第二步：准备 install.sh（宿主机安装）

参考现有工具的 `install.sh`，使用统一模板：

```bash
#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="my_new_env"
YML_FILE="$SCRIPT_DIR/environment.yml"

if conda env list | grep -q "^${ENV_NAME} \|^${ENV_NAME}$"; then
    read -p "环境已存在，是否重装? (y/N) " answer
    [[ "$answer" != "y" && "$answer" != "Y" ]] && exit 0
    conda env remove -n "${ENV_NAME}" -y
fi

conda env create -f "${YML_FILE}"
echo "✅ 安装完成: conda activate ${ENV_NAME}"
```

同时在 `install_all.sh` 的 `TOOLS` 数组中追加工具名。

---

### 第三步：在 Dockerfile 中添加对应内容

当前 Dockerfile 分为两处需要修改：

**① 在第 4 节（COPY yml 文件区）末尾追加：**
```dockerfile
# my_new_tool
COPY additional_tools/server/tools/my_new_tool/environment.yml /tmp/envs/my_new_tool.yml
```

**② 在第 5 节（创建 conda 环境区）末尾追加对应 `RUN` 块：**

**方式 A：从 environment.yml 创建（推荐）**
```dockerfile
# ─── 环境 N: my_new_env（工具描述）────────────────────────────────────────────
RUN conda env create -n my_new_env -f /tmp/envs/my_new_tool.yml \
    && conda clean -afy
```
> 若 `environment.yml` 内的 `name:` 字段与目标名不同，`-n my_new_env` 会自动覆盖，无需修改 yml 文件。

**方式 B：从 requirements.txt 安装**
```dockerfile
# ─── 环境 N: my_new_env ────────────────────────────────────────────────────────
RUN conda create -n my_new_env python=3.10 -y \
    && conda run -n my_new_env pip install --no-cache-dir \
        -r /app/server/tools/my_new_tool/requirements.txt \
    && conda clean -afy
```

**方式 C：直接指定包**
```dockerfile
# ─── 环境 N: my_new_env ────────────────────────────────────────────────────────
RUN conda create -n my_new_env python=3.11 -y \
    && conda run -n my_new_env pip install --no-cache-dir torch rdkit \
    && conda clean -afy
```

**③ 若工具需要 pip install 本地源码（如 rxnflow），在第 6 节首部追加：**
```dockerfile
COPY additional_tools/server/tools/my_new_tool/my_lib /app/server/tools/my_new_tool/my_lib
RUN conda run -n my_new_env pip install --no-cache-dir \
        -e /app/server/tools/my_new_tool/my_lib \
    && conda clean -afy
```

---

### 第四步：如果新工具需要 GPU

1. `config.json` 中设置 `"gpu": true`（运行时自动分配 GPU，无需改 Dockerfile）
2. 确认 Dockerfile 基础镜像的 CUDA 版本与工具 PyTorch 版本兼容：

| PyTorch 版本 | 推荐基础镜像 |
|---|---|
| 2.6.x | `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` |
| 2.5.x | `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` |
| 2.4.x | `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` |
| 2.1.x | `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` |

> conda 环境中的 PyTorch 会自带 CUDA runtime，无需基础镜像版本完全匹配；`12.1` 基础镜像可兼容运行 `11.8` 的 conda 环境。

---

## 构建镜像

```bash
# 必须在 CAi/ 目录下执行
cd /path/to/CAi

# 完整重新构建
docker build -t tool-server -f additional_tools/server/Dockerfile .

# 指定代理（实验室网络环境）
docker build \
    --build-arg HTTP_PROXY=http://proxy:port \
    --build-arg HTTPS_PROXY=http://proxy:port \
    -t tool-server \
    -f additional_tools/server/Dockerfile .

# 或用 docker compose（推荐）
docker compose -f additional_tools/server/docker-compose.yml up --build -d
```

---

## 运行时环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `TOOL_SERVER_HOST` | `0.0.0.0` | 监听地址 |
| `TOOL_SERVER_PORT` | `8001` | 监听端口 |
| `GPU_IDS` | 自动检测 | 允许使用的显卡编号，如 `0,1` |
| `JOB_MAX_AGE_DAYS` | `7` | Job 目录保留天数 |

```bash
# 示例：指定 GPU 0 和 1，端口改为 9000
docker run --gpus all \
    -e GPU_IDS=0,1 \
    -e TOOL_SERVER_PORT=9000 \
    -p 9000:9000 \
    tool-server
```

---

## 宿主机直接安装（不用 Docker）

每个工具都有独立的 `install.sh`，可单独安装或通过顶层脚本一键安装：

```bash
cd additional_tools/server

# 安装所有工具
./install_all.sh

# 只安装指定工具
./install_all.sh vina scscore

# 全部安装，跳过确认提示
./install_all.sh -y

# 单独安装某个工具
bash tools/vina/install.sh
```

---

## 当前工具清单

| 工具名 | conda 环境 | GPU | 依赖来源 |
|---|---|---|---|
| `scscore` | `scscore` | ✗ | `tools/scscore/environment.yml`（含 fastapi/uvicorn，承载主进程） |
| `vina` | `vina` | ✗ | `tools/vina/environment.yml` |
| `libinvent` | `lib-invent` | ✗ | `tools/libinvent/environment.yml` |
| `pmic` | `gflow312` | ✓ | `tools/pmic/environment.yml`（yml 内 name=chemprop，用 `-n` 覆盖） |
| `scaffold` | `chempro310` | ✓ | `tools/scaffold/environment.yml` |
| `toxicity` | `toxicity` | ✓ | `tools/toxicity/environment.yml` |
| `rxnflow` | `rxnflow` | ✓ | `tools/rxnflow/environment.yml` + `pip install -e RxnFlow/` |
| `reinvent4` | `reinvent4` | ✓ | `conda create python=3.10` + `pip install reinvent4`（PyPI） |
| `drugex` | `drugex` | ✓ | `conda create python=3.10` + pip from GitHub |

> **注意**：`reinvent4` 和 `drugex` 在 Docker 构建时需要访问外网（PyPI / GitHub）。若构建环境无网络，需提前打包 wheel 文件并改用本地安装。
