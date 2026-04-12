# A1pro 解决方案总结

## 你的需求

> "我需要让 agent 每次都自动加载 additional_tools 模块的工具，而不是需要手动加载。主要是工具依赖都比较多。"

## 解决方案

我为你创建了 **A1pro** - 一个扩展的 Biomni Agent 类，具有以下特性：

### ✅ 核心功能

1. **自动工具加载**
   - 初始化时自动扫描并加载 `additional_tools` 模块中的所有工具
   - 无需手动调用 `add_tool()`
   - 支持复杂依赖的工具

2. **灵活的加载模式**
   - 立即加载（默认）
   - 延迟加载（lazy loading）
   - 手动排除特定工具

3. **工具管理**
   - 列出所有可用工具
   - 查看工具详细信息
   - 热更新（重新加载工具）
   - 从文件加载特定工具

---

## 文件结构

```
biomiplus/
├── bio_agent/
│   └── agent.py                    # ✅ A1pro 类实现
├── additional_tools/
│   ├── __init__.py                 # ✅ 工具模块初始化
│   └── example_tools.py            # ✅ 示例工具（7个）
├── README.md                       # ✅ 完整文档
├── QUICKSTART.md                   # ✅ 快速入门
└── test_a1pro.py                   # ✅ 测试套件

项目根目录/
├── complex_tool_dependencies_guide.md  # ✅ 复杂依赖处理指南
└── A1PRO_SOLUTION_SUMMARY.md          # ✅ 本文档
```

---

## 使用方法

### 基本使用（3 步）

```python
# 1. 在 additional_tools/ 中创建你的工具
# biomiplus/additional_tools/my_tools.py
def my_tool(param: str) -> dict:
    """工具描述"""
    import pandas as pd  # 复杂依赖在函数内导入
    # 工具逻辑
    return result

# 2. 初始化 A1pro（自动加载所有工具）
from biomiplus.bio_agent.agent import A1pro

agent = A1pro(
    llm="your-model",
    source="Custom",
    base_url="your-api-url",
    api_key="your-api-key",
    use_tool_retriever=False,
    expected_data_lake_files=[]
)
# 输出：✅ 成功加载 X/X 个工具

# 3. 使用工具
response = agent.go("使用 my_tool 处理数据")
```

### 高级配置

```python
# 排除某些工具
agent = A1pro(
    llm="your-model",
    exclude_tools=["tool1", "tool2"]
)

# 延迟加载（节省初始化时间）
agent = A1pro(
    llm="your-model",
    lazy_load=True
)

# 自定义工具模块路径
agent = A1pro(
    llm="your-model",
    tools_module="my_custom_tools"
)
```

---

## 如何处理复杂依赖？

### 核心原则：在函数内部导入

```python
def complex_tool(param: str) -> dict:
    """有复杂依赖的工具"""
    
    # ✅ 在函数内部导入所有依赖
    import pandas as pd
    import numpy as np
    from rdkit import Chem
    from my_custom_module import MyClass
    
    # 工具逻辑
    result = process_with_dependencies(param)
    
    return result
```

### 为什么这样可行？

1. **持久化命名空间**
   - Biomni 使用 `_persistent_namespace` 保存所有执行状态
   - 导入的模块在多次执行间保持

2. **Agent 生成正确的代码**
   - Agent 看到工具 schema 后，会生成包含导入的代码
   - 执行时，函数内的 import 会自动执行

3. **自动注入**
   - A1pro 自动将工具注入到执行环境
   - 工具可以直接调用，无需额外配置

---

## 示例工具

### 1. 简单工具

```python
def simple_calculator(expression: str) -> float:
    """计算数学表达式"""
    return eval(expression)
```

### 2. 第三方库依赖

```python
def fetch_molecule_data(molecule_name: str) -> dict:
    """从 PubChem 获取分子信息"""
    import requests
    
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{molecule_name}/JSON"
    response = requests.get(url)
    return response.json()
```

### 3. 复杂依赖（RDKit）

```python
def smiles_to_properties(smiles: str) -> dict:
    """使用 RDKit 计算分子性质"""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    mol = Chem.MolFromSmiles(smiles)
    
    return {
        'molecular_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        # ...
    }
```

### 4. 调用其他工具

```python
def batch_analysis(smiles_list: list[str]) -> dict:
    """批量分析多个分子"""
    results = []
    
    for smiles in smiles_list:
        # 调用其他工具
        result = smiles_to_properties(smiles)
        results.append(result)
    
    return {'results': results}
```

---

## 工具管理命令

```python
# 列出所有工具
agent.list_available_tools()

# 查看工具详情
agent.get_tool_info("tool_name")

# 重新加载工具（热更新）
agent.reload_tools()

# 从文件加载工具
agent.add_tool_from_file("path/to/tools.py")
agent.add_tool_from_file("path/to/tools.py", "specific_function")
```

---

## 测试

运行测试套件：

```bash
python biomiplus/test_a1pro.py
```

测试包括：
1. 基本初始化
2. 简单工具使用
3. 分子数据工具
4. 排除工具
5. 工具信息查看
6. 热更新
7. 自定义文件加载

---

## 优势

### 相比手动 `add_tool()`

| 特性 | 手动 add_tool() | A1pro 自动加载 |
|------|----------------|----------------|
| 添加新工具 | 需要修改代码 | 只需创建函数 |
| 初始化时间 | 每次都要写 | 自动完成 |
| 工具管理 | 手动追踪 | 自动管理 |
| 热更新 | 不支持 | 支持 reload() |
| 工具列表 | 手动维护 | 自动生成 |

### 处理复杂依赖

| 依赖类型 | 支持 | 说明 |
|---------|------|------|
| 第三方库 | ✅ | pandas, numpy, requests 等 |
| 化学库 | ✅ | RDKit, OpenBabel 等 |
| 生物库 | ✅ | Biopython, scanpy 等 |
| 自定义模块 | ✅ | 你自己的模块 |
| 大型模型 | ✅ | 延迟加载支持 |
| API 调用 | ✅ | 任何 HTTP API |

---

## 最佳实践

### 1. 工具组织

```
additional_tools/
├── __init__.py
├── chemistry_tools.py      # 化学相关
├── biology_tools.py        # 生物相关
├── data_tools.py           # 数据分析
└── api_tools.py            # API 调用
```

### 2. 工具命名

```python
# ✅ 好的命名
def fetch_molecule_data(...)
def analyze_protein_sequence(...)
def calculate_binding_affinity(...)

# ❌ 避免的命名
def get_data(...)
def process(...)
def tool1(...)
```

### 3. 错误处理

```python
def robust_tool(param):
    try:
        import some_library
    except ImportError:
        return {
            'error': 'some_library 未安装',
            'hint': '请运行: pip install some_library'
        }
    
    try:
        result = process(param)
        return {'success': True, 'data': result}
    except Exception as e:
        return {'success': False, 'error': str(e)}
```

### 4. 文档字符串

```python
def my_tool(param1: str, param2: int) -> dict:
    """
    工具的简短描述（显示在工具列表）
    
    详细描述工具的功能...
    
    Parameters:
        param1: 参数1描述
        param2: 参数2描述
    
    Returns:
        返回值描述
    
    Examples:
        >>> my_tool("test", 42)
        {'result': 'success'}
    """
```

---

## 常见问题

### Q: 工具没有被加载？

**检查**：
1. 函数名不能以 `_` 开头
2. 函数必须在 `additional_tools` 模块中
3. 查看初始化时的输出信息

### Q: 依赖库没有安装？

**解决**：
```python
def my_tool(param):
    try:
        import required_library
    except ImportError:
        return {
            'error': 'required_library 未安装',
            'hint': '请运行: pip install required_library'
        }
```

### Q: 如何调试工具？

**方法**：
```python
# 1. 直接测试函数
from biomiplus.additional_tools.my_tools import my_tool
result = my_tool("test")

# 2. 查看工具信息
agent.get_tool_info("my_tool")

# 3. 热更新
agent.reload_tools()
```

---

## 下一步

1. ✅ 阅读 `biomiplus/README.md` - 完整文档
2. ✅ 阅读 `biomiplus/QUICKSTART.md` - 快速入门
3. ✅ 查看 `additional_tools/example_tools.py` - 示例工具
4. ✅ 运行 `python biomiplus/test_a1pro.py` - 测试
5. ✅ 开始创建你自己的工具！

---

## 总结

A1pro 为你提供了：

1. **自动化**：无需手动 `add_tool()`，自动加载所有工具
2. **灵活性**：支持排除、延迟加载、热更新
3. **强大**：处理任意复杂的依赖
4. **易用**：3 步即可开始使用
5. **可维护**：清晰的项目结构和文档

**你只需要**：
1. 在 `additional_tools/` 中创建工具函数
2. 在函数内部导入依赖
3. 初始化 A1pro

**A1pro 会自动处理其余部分！** 🚀
