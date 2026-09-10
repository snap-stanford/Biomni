# Memory System Benchmark

对比 **Baseline Memory System** 与 **Improved Memory System** 的实验框架。

本目录目前只包含**准备阶段**产物（目录结构、case 模板、语料、基线分析、人工标注指南），
**尚未运行任何 Benchmark**。

## 目录结构

```
evaluation/memory_benchmark/
├── README.md                    # 本文件
├── baseline_analysis.md         # Baseline vs Improved 代码差异分析
├── HUMAN_REVIEW.md              # Ground Truth 人工标注指南 + Summary Quality 评分标准
├── memory_corpus.json           # 共享语料：memory_key -> (user_id, trace_ref)
├── ground_truth/                # 人工确认后的 Ground Truth（当前为空）
├── extraction/                  # Extraction Benchmark：trace -> summary + facts
│   └── extraction_cases.json    # 20 个 extraction case（expected_* 待人工标注）
├── retrieval/                   # Retrieval Benchmark：query -> 召回
│   ├── retrieval_cases.json     # 20 个 retrieval case（expected_memory_keys 待人工标注）
│   └── memory_id_map.template.json  # runner 运行时生成的 memory_key -> memory_id 映射模板
├── continuation/                # Continuation Benchmark：连续任务
│   └── continuation_cases.json  # 15 个 continuation scenario
└── results/                     # 跑分结果（当前为空）
```

## 关键约束（务必遵守）

1. **Ground Truth 只能由人工确认。** 模板中的 `expected_facts`、
   `expected_summary_information`、`expected_memory_keys` 等字段当前为「待人工标注」，
   **禁止**把 LLM 自动生成的答案直接当 Ground Truth。标注方法见 `HUMAN_REVIEW.md`。
2. **不修改生产代码。** 本目录只做评测，不改 `memory/*.py`、`database/*.py`、
   Agent 生产代码。
3. **不通过改配置人为提高结果。** Baseline 与 Improved 必须跑在同一
   embedding / 同一 LLM / 同一数据下，否则对比无意义。

## memory_key 机制（核心设计）

**问题**：Memory 的 `memory_id` 是 `create_memory` 返回的 UUID，每次 ingest 随机生成，
跨版本不可比，无法直接用作 Ground Truth。

**方案**：Ground Truth 引用**稳定的语义标识 `memory_key`**，真实 `memory_id` 由 runner
在**每个版本运行时动态映射**。

```
memory_corpus.json           （共享、跨版本不变）
  └─ memory_key ─ user_id ─ trace_ref（指向 extraction case 的 trace）
             │
             │  ① 每个版本各自 ingest 这份语料（MemorySystem.ingest(trace)）
             │     create_memory 返回该版本下的 memory_id(UUID)
             ▼
results/<version>/memory_id_map.json   （per-version，运行时生成）
  └─ { "memory_key": "<该版本下的 memory_id>" }
             │
             │  ② 检索：episodic.search_memory(query, user_id, k) -> top-k memory_id
             │     反向映射回 memory_key，与 expected_memory_keys 比对
             ▼
Recall@K / Precision@K 按 memory_key 判定（与 UUID 无关，跨版本可比）
```

- Baseline 写 `results/baseline/memory_id_map.json`，Improved 写
  `results/improved/memory_id_map.json`，模板见 `retrieval/memory_id_map.template.json`。
- **Ground Truth 文件里绝不出现 UUID**，只出现 `memory_key`。

## 三个 Benchmark 的目标

| Benchmark | 输入 | 输出 | 主要度量 |
|-----------|------|------|----------|
| Extraction | 一条 agent trace | summary + facts | Fact Accuracy（Precision/Recall of facts）、Summary Quality（6 维度 0–12，见 HUMAN_REVIEW.md） |
| Retrieval | 一条 query + user_id | 召回的 memory/facts | Recall@K、Precision@K（按 memory_key 判定）、user 隔离 |
| Continuation | session A（任务）→ session B（继续） | session B 是否正确接续 | Continuation Success（具体可判定行为） |

### 规模

- Extraction：**20** cases（`ext_001`…`ext_020`）。
- Retrieval：**20** cases，覆盖 7 类查询：exact_lexical / paraphrase /
  low_lexical_overlap / specific_fact / previous_task / multi_memory / user_isolation。
- Continuation：**15** scenarios，覆盖 10 类：continue_previous_analysis /
  reuse_specific_value / reuse_specific_conclusion / reuse_tool_result /
  reuse_selected_object / know_last_step / know_completed_work / know_incomplete_work /
  cross_memory_continuation / user_isolation。

### 关于 Retrieval 的度量层级（重要）

`baseline_analysis.md` 已指出：`embedding_provider` 默认仍是 `hash`（非语义化），
Baseline 与 Improved 完全相同。因此：

- **Recall@K（memory 级）**：建议直接调用 `episodic.search_memory(query, user_id, k)`
  拿 top-k memory_id，映射回 memory_key 比对。hash embedding 下 Baseline ≈ Improved，
  换真实 embedding 后才可能看到差异。
- **Precision@K（fact 级）**：调用 `retriever.retrieve(query, user_id)` 拿 facts，
  看 Improved 的去重 / active 过滤 / 排序是否让 fact 更干净。这是 Improved 的**主要优势面**。
- `low_lexical_overlap` 类 query 是专门为真实 embedding 设计的：hash embedding 下预期失败，
  这本身就是一个有效发现（证明 hash embedding 的语义召回局限）。

## 下一步（未开始）

1. 人工按 `HUMAN_REVIEW.md` 标注 `extraction` / `retrieval` / `continuation` 的
   `expected_*` 字段，确认后归档到 `ground_truth/`。
2. 实现 runner（ingest 语料 → 生成 `memory_id_map` → 跑三个 benchmark），
   固定 Baseline（commit `4349ab3`）与 Improved（当前 working tree）两个版本。
3. 在同一环境、同一 embedding / LLM / 数据下跑同一套 case，结果写入 `results/`。
