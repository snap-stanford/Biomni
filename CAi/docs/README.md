# A1pro - 增强版 Biomni Agent

## 概述

A1pro 是 Biomni A1 Agent 的增强版本，主要特性是**自动加载 `additional_tools` 模块中的所有工具**。

## 核心特性

### 1. 自动工具加载
- ✅ 自动扫描并加载 `additional_tools` 模块中的所有工具函数
- ✅ 支持复杂依赖（第三方库、自定义模块）
- ✅ 支持工具热更新（开发模式）
- ✅ 支持手动排除特定工具

### 2. 灵活的加载模式
- **立即加载**（默认）：初始化时加载所有工具
- **延迟加载**：首次使用时才加载工具
- **手动加载**：从文件加载特定工具

### 3. 工具管理
- 列出所有可用工具
- 查看工具详细信息
- 重新加载工具（热更新）
- 排除特定工具

---

## 快速开始

### 基本使用

```python
from biomiplus.bio_agent.agent import A1pro

# 1. 初始化 A1pro（自动加载所有工具）
agent = A1pro(
    llm="claude-sonnet-4-5-20250929",
    source="Custom",
    base_url="http://your-api-url/v1/",
    api_key="your-api-key",
    use_tool_retriever=False,
    expected_data_lake_files=[]
)

# 2. 查看已加载的工具
agent.list_available_tools()

# 3. 使用工具
response = agent.go("请使用 fetch_molecule_data 获取阿司匹林的分子信息")
print(response)
```

### 高级配置

```python
# 排除某些工具
agent = A1pro(
    llm="your-model",
    exclude_tools=["tool1", "tool2"],  # 不加载这些工具
    auto_load_tools=True
)

# 延迟加载模式（节省初始化时间）
agent = A1pro(
    llm="your-model",
    lazy_load=True  # 工具在首次使用时才加载
)

# 自定义工具模块路径
agent = A1pro(
    llm="your-model",
    tools_module="my_custom_tools"  # 从其他模块加载
)
```

---

## 编写工具

### 工具编写规范

在 `additional_tools/` 目录下创建 Python 文件，每个工具都是一个函数：

```python
# additional_tools/my_tools.py

def my_awesome_tool(param1: str, param2: int) -> dict:
    """
    工具的简短描述（这会显示在工具列表中）

    详细描述工具的功能和用途...

    Parameters:
        param1: 参数1的描述
        param2: 参数2的描述

    Returns:
        返回值的描述

    Examples:
        >>> my_awesome_tool("test", 42)
        {'result': 'success'}
    """
    # 在函数内部导入依赖
    import pandas as pd
    import requests
    from my_custom_module import helper_function

    # 工具逻辑
    result = helper_function(param1, param2)

    return {'result': result}
```

### 关键要点

1. **完整的文档字符串**
   - 第一行：简短描述（显示在工具列表）
   - Parameters：参数说明
   - Returns：返回值说明
   - Examples：使用示例（可选）

2. **类型注解**
   ```python
   def tool(param: str, count: int = 5) -> dict:
   ```

3. **在函数内部导入依赖**
   ```python
   def tool(param):
       import pandas as pd  # ✅ 在函数内导入
       import my_module
       # ...
   ```

4. **避免的做法**
   - ❌ 不要使用下划线开头的函数名（会被跳过）
   - ❌ 不要依赖全局变量
   - ❌ 不要使用相对导入

---

## 工具示例

### 示例 1：简单工具

```python
def simple_calculator(expression: str) -> float:
    """简单计算器 - 计算数学表达式"""
    return eval(expression)
```

### 示例 2：第三方库依赖

```python
def fetch_molecule_data(molecule_name: str) -> dict:
    """获取分子数据 - 从 PubChem 获取分子信息"""
    import requests

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{molecule_name}/JSON"
    response = requests.get(url)
    return response.json()
```

### 示例 3：复杂依赖（RDKit）

```python
def smiles_to_properties(smiles: str) -> dict:
    """SMILES 转化学性质 - 使用 RDKit 计算分子性质"""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles)

    return {
        'molecular_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        # ...
    }
```

### 示例 4：调用其他工具

```python
def batch_analysis(smiles_list: list[str]) -> dict:
    """批量分析 - 分析多个分子"""
    results = []

    for smiles in smiles_list:
        # 调用其他工具
        result = smiles_to_properties(smiles)
        results.append(result)

    return {'results': results}
```

---

## 工具管理

### 列出所有工具

```python
agent.list_available_tools()
```

输出：
```
============================================================
📋 A1pro 可用工具列表
============================================================

🔧 Biomni 内置工具:
  [biomni.tool.genetics]
    - query_gene_info
    - get_gene_sequence
    - ...

🎯 Additional Tools (5 个):
  - simple_calculator: 简单计算器 - 计算数学表达式
  - fetch_molecule_data: 获取分子数据 - 从 PubChem 获取分子信息
  - smiles_to_properties: SMILES 转化学性质 - 使用 RDKit 计算分子性质
  - ...
============================================================
```

### 查看工具详情

```python
agent.get_tool_info("fetch_molecule_data")
```

### 重新加载工具（热更新）

```python
# 修改 additional_tools 中的代码后
agent.reload_tools()
```

### 从文件加载工具

```python
# 加载特定文件中的所有工具
agent.add_tool_from_file("path/to/my_tools.py")

# 加载特定函数
agent.add_tool_from_file("path/to/my_tools.py", "specific_function")
```

---

## 项目结构

```
biomiplus/
├── bio_agent/
│   └── agent.py              # A1pro 类定义
├── additional_tools/
│   ├── __init__.py           # 工具模块初始化
│   ├── example_tools.py      # 示例工具
│   ├── chemistry_tools.py    # 化学工具（你的工具）
│   ├── biology_tools.py      # 生物工具（你的工具）
│   └── data_tools.py         # 数据工具（你的工具）
└── README.md                 # 本文档
```

---

## 常见问题

### Q1: 工具没有被加载？

**检查清单**：
1. 函数名不能以下划线开头
2. 函数必须在 `additional_tools` 模块中
3. 检查是否在 `exclude_tools` 列表中
4. 查看初始化时的输出信息

### Q2: 工具依赖的库没有安装？

在工具函数中添加友好的错误提示：

```python
def my_tool(param):
    try:
        import some_library
    except ImportError:
        return {
            'error': 'some_library 未安装',
            'hint': '请运行: pip install some_library'
        }
    # ...
```

### Q3: 如何处理复杂的初始化逻辑？

使用工厂函数模式：

```python
def create_database_tool(db_path: str):
    """工厂函数：创建数据库工具"""
    # 初始化连接（只执行一次）
    import sqlite3
    conn = sqlite3.connect(db_path)

    def query_database(sql: str) -> dict:
        """查询数据库"""
        import pandas as pd
        result = pd.read_sql_query(sql, conn)
        return result.to_dict()

    return query_database

# 使用
db_tool = create_database_tool('/path/to/db.sqlite')
agent.add_tool(db_tool)
```

### Q4: 如何组织大量工具？

按功能分类到不同文件：

```python
# additional_tools/__init__.py
from .chemistry_tools import *
from .biology_tools import *
from .data_tools import *
```

### Q5: 开发时如何快速测试工具？

```python
# 1. 修改工具代码
# 2. 重新加载
agent.reload_tools()

# 3. 测试
response = agent.go("测试我的新工具")
```

---

## 最佳实践

### 1. 工具命名
- ✅ 使用描述性的名称：`fetch_molecule_data`
- ❌ 避免模糊的名称：`get_data`

### 2. 错误处理
```python
def robust_tool(param):
    try:
        # 主要逻辑
        result = process(param)
        return {'success': True, 'data': result}
    except Exception as e:
        return {'success': False, 'error': str(e)}
```

### 3. 返回结构化数据
```python
# ✅ 好的做法
return {
    'status': 'success',
    'data': {...},
    'metadata': {...}
}

# ❌ 避免只返回字符串
return "Result: 42"
```

### 4. 添加使用示例
```python
def my_tool(param: str) -> dict:
    """
    工具描述

    Examples:
        >>> my_tool("test")
        {'result': 'success'}

        >>> my_tool("error")
        {'error': 'Invalid input'}
    """
```

---

## 性能优化

### 延迟加载大型依赖

```python
def heavy_tool(param):
    """使用大型库的工具"""
    # 只在需要时导入
    if param == "use_heavy_lib":
        import heavy_library
        return heavy_library.process(param)
    else:
        # 简单处理
        return simple_process(param)
```

### 缓存计算结果

```python
_cache = {}

def cached_tool(param):
    """带缓存的工具"""
    if param in _cache:
        return _cache[param]

    result = expensive_computation(param)
    _cache[param] = result
    return result
```

---

## 贡献指南

欢迎添加新工具！请遵循：

1. 在 `additional_tools/` 中创建新文件或添加到现有文件
2. 遵循工具编写规范
3. 添加完整的文档字符串
4. 测试工具功能
5. 更新 README（如果需要）

---

## 许可证

与 Biomni 项目相同
