# Baseline Analysis: Memory System 前后差异

> 本文件用于记录「Improved Memory System 相比 Baseline 到底改了什么」，以及
> 每一类改动**理论上**应该影响哪些评测指标、哪些不该影响。
> 这是后续 Benchmark 假设（hypothesis）的依据。

---

## 1. 版本定位

| 名称 | 位置 |
|------|------|
| **BASELINE** | commit `4349ab3` — `feat: add production memory system`（2026-08-27 21:14:48 +0800） |
| **IMPROVED** | 当前 working tree = `4349ab3` + **已暂存改动（git index）** + 1 处未暂存修复 |

### Baseline 为什么可靠

- `git log --oneline --all -- memory/` 只有一条：`4349ab3 feat: add production memory system`。
- 该 commit 的父提交是 `400c1f3`（= `origin/main`），父提交里**完全没有** `memory/`、`database/` 目录。
- 也就是说：Memory 系统整体由 `4349ab3` 一次性引入，它是「本轮完善工作开始之前」唯一已提交的版本。
- 之后的完善工作（lifecycle / scoring / feedback / user 隔离 / tests）**全部是未提交的 staged 改动**，落在 working tree 上。

因此 **Baseline = `4349ab3`，Improved = 当前 working tree**，判定无歧义。

> 备注：`git status` 里仓库几乎所有文件显示为全量改动，那是行尾/文件模式差异（噪声），
> 与 Memory 工作无关。真正的 Memory 改动只集中在 `memory/`、`database/`、`tests/`。

---

## 2. 文件级变化（Baseline → Improved）

| 文件 | 变化量（staged） | 性质 |
|------|------------------|------|
| `memory/scoring.py` | +89 行 | **新增**：importance/recency/usage/feedback 纯函数 |
| `tests/test_memory_lifecycle.py` | +820 行 | **新增**：46 个测试 |
| `memory/semantic.py` | +418 行 | 大规模扩展 |
| `memory/retriever.py` | +132 行 | 大规模扩展 |
| `memory/system.py` | +78 行 | 扩展 |
| `memory/models.py` | +58 行 | 扩展 |
| `memory/prompts.py` | +50 行 | 扩展（summary 结构） |
| `database/models.py` | +40 行 | 扩展（Fact 生命周期列） |
| `database/migrations.py` | +51 行 | 扩展（列迁移） |
| `memory/vector.py` | +41 行 | 扩展（where 过滤、huggingface） |
| `memory/validator.py` | +33 行 | 小重构 |
| `memory/episodic.py` | +10 行 | 扩展（user 过滤） |
| `memory/__init__.py` | +6 行 | 导出 scoring |
| `memory/extractor.py` | **0（未变）** | 提取管线不变 |
| `memory/working.py` | **0（未变）** | 工作记忆不变 |

**关键结论：`extractor.py` 在 Baseline 与 Improved 之间完全一致。**
也就是说「trace → 结构化输出」的**机械管线**没变，变化的只是：
- 提取的**提示词**（`prompts.py`）
- 提取之后**如何存储/去重/排序/过滤/隔离**（`semantic.py`、`retriever.py`、`vector.py`）

---

## 3. 逐项差异

### 3.1 Memory Formation（记忆形成）

| 维度 | Baseline | Improved | 差异 |
|------|----------|----------|------|
| summary 要求 | 「concise, self-contained summary… include key parameters, tools used, results」 | 明确 `(a) Task/Context → (b) Tools/Key Parameters → (c) <observation> 实际结果 → (d) <solution> 最终结论 → (e) Task Status/Remaining Work`，并禁止「executed successfully」、要求区分 observation 与 solution、失败也要记原因 | **Prompt 结构化，直接指向 Summary Quality** |
| fact 提取 | entity/relation/value + confidence + source | 同左（规则不变） | 无变化 |
| fact 校验 | confidence ≥ min、entity/value 非空、source 非 `llm` | 同左（逻辑等价，仅重构出 `_rejection_reason` 便于日志） | 无实质变化 |

### 3.2 Memory Storage / Reliability（存储与可靠性，集中在 `semantic.py`）

| 能力 | Baseline | Improved |
|------|----------|----------|
| 去重（同 entity+relation+value） | ❌ 直接插入，重复行 | ✅ 复用行 + bump confidence |
| 冲突（单值 relation supersede / 多值共存） | ❌ | ✅ `SINGLE_VALUE_RELATIONS` |
| Fact 生命周期（active/superseded/retracted/expired） | ❌ 无 status 字段 | ✅ 完整状态机 |
| 用户反馈（feedback + 阈值 retract） | ❌ | ✅ `update_fact_feedback` |
| 过期（TTL，基于 `created_at`） | ❌ | ✅ `expire_facts` |
| access_count 计数 | ❌ | ✅ 仅对进入 context 的 fact 计数 |
| SQL 层 user 隔离 | ❌ `get_facts_by_memory` 无 user 校验 | ✅ `get_active_facts_by_memories` 二次校验 `Memory.user_id` + 批量查询（无 N+1） |
| 批量写入 + 重试 | ❌ 逐条 commit | ✅ 分块 + 指数退避 + 降级逐条 |
| 清理（cleanup） | ❌ | ✅ 删无 active fact 的 memory + vector 先删 |

### 3.3 Memory Retrieval（检索，`retriever.py` + `episodic.py` + `vector.py`）

| 能力 | Baseline | Improved |
|------|----------|----------|
| user 隔离（向量层） | ❌ `search_memory(query, k)` 无 where | ✅ `search_memory(query, user_id, k)` + `where={"user_id":...}` |
| 只返回 active fact | ❌ 返回该 memory 的**所有** fact（含过期/被取代） | ✅ 只取 active |
| 排序 | ❌ 无排序，按命中顺序全量 append | ✅ 两阶段：cold-start 按 confidence、mature 按 importance_score |
| 数量上限 | ❌ 无上限 | ✅ `max_facts` cap |
| 批量取 fact | ❌ 每个 memory 一条查询 | ✅ 一条 SQL 批量取 |
| importance_score | ❌ 不存在 | ✅ 计算并附加到 fact |

### 3.4 数据模型（`database/models.py` + `memory/models.py`）

| 字段 | Baseline | Improved |
|------|----------|----------|
| `Fact.created_at` / `updated_at` | ❌ | ✅ |
| `Fact.status` | ❌ | ✅ |
| `Fact.access_count` | ❌ | ✅ |
| `Fact.positive/negative_feedback_count` | ❌ | ✅ |
| `MemoryFact` 上的 `created_at/updated_at/status/importance_score` | ❌ | ✅ |
| `MemoryConfig` 的 TTL / 权重 / max_facts / feedback 阈值等 | ❌ | ✅（含权重和=1.0 校验） |

---

## 4. 理论影响矩阵（关键）

> 这是 Benchmark 的核心假设：哪些改动**应该**体现在某个指标上，哪些**不该**。

### 4.1 应直接影响评测质量的改动

| 改动 | 预期影响 |
|------|----------|
| **prompts.py 的 (a)–(e) 结构** | **Summary Quality ↑**（覆盖率、可续接性） |
| **去重 + supersede + 只返回 active** | **Fact Accuracy ↑**（Precision：更少重复/过期 fact）、**Precision@K ↑** |
| **两阶段排序 + max_facts cap** | **Precision@K ↑**、**Recall@K**（重要 fact 排前） |
| **user 隔离（向量 + SQL 双层）** | **Precision@K ↑**（不混入他人 memory）、**Continuation Success ↑** |
| **feedback/retract** | 长期 **Fact Accuracy ↑**（错误 fact 被收回） |

### 4.2 只属于 reliability / safety，**不应**直接影响 retrieval quality 的改动

| 改动 | 为什么不该影响评测指标 |
|------|------------------------|
| `created_at`/`updated_at` 语义修复（access_count / expire 不 bump updated_at） | 数据完整性；与召回排序无直接关系 |
| TTL 过期、cleanup 幂等、vector 删除失败时保护 SQL | 存储卫生；除非 benchmark 特意测「过期后是否被删」 |
| 批量写入 + 重试 | ingestion 可靠性；不改变最终落库内容 |
| `_add_missing_columns` 迁移 | 老库兼容；新库无差异 |
| `expire_facts` 的 updated_at pin | 同上，数据完整性 |

### 4.3 一个必须注意的「不变项」

**`embedding_provider` 默认仍是 `hash`（非语义化），Baseline 与 Improved 完全相同。**

- 因此**向量层的相似度质量两边一致**：`top_k` 个 memory 的**召回顺序在两边相同**。
- Improved 的优势全部来自「向量召回**之后**」的 fact 过滤 / 排序 / 去重 / 隔离。
- 推论：
  - **Recall@K（memory 级别）在 hash embedding 下 Baseline ≈ Improved**（看不出差异）。
  - 要测出 Retrieval 的真实差异，要么换真实 embedding（DeepSeek 无 embedding，需另配），
    要么把 Retrieval 指标聚焦在 **fact 级别** 的 Precision（Improved 会明显更干净）。

---

## 5. 对 Benchmark 设计的直接启示

1. **Extraction Benchmark**：主要看 prompts.py 改动带来的 **Summary Quality** 提升。
   Fact 提取规则没变，所以 Fact Accuracy 的差异预期**较小**（主要来自去重/校验的边际影响）。
2. **Retrieval Benchmark**：应把 **Precision@K（fact 级）** 作为主指标，Recall@K（memory 级）
   在 hash embedding 下预期**无差异**——除非换真实 embedding。
3. **Continuation Benchmark**：应同时收益于 Summary Quality 提升 + user 隔离 + active 过滤，
   是最能体现「Improved 整体更好」的场景。
4. **必须控制变量**：Baseline 与 Improved 必须用同一 embedding、同一 LLM、同一数据、
   同一 user_id 语料；否则任何指标差异都无法归因。
