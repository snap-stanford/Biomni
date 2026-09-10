# Benchmark 结果：Baseline vs Improved（Differentiated benchmark）

> 完整三组件最终结果见 [FINAL.md](FINAL.md)。本文件只记录 `run_benchmark.py` 这一组件。

运行方式（版本无关，同一 venv `<benchmark-venv>`，同一 SQLite/Chroma/hash embedding，
每个 benchmark section 用独立的临时 DB + Chroma collection 隔离，避免跨 section 状态污染）：

```bash
PYTHONPATH=<repo-root>         venv/bin/python3 run_benchmark.py > results/improved.json
PYTHONPATH=<baseline-checkout> venv/bin/python3 run_benchmark.py > results/baseline.json
venv/bin/python3 compare.py results/baseline.json results/improved.json
```

## 对比表

| metric | baseline | improved | delta |
|--------|---------:|---------:|------:|
| **Conflict Resolution** | | | |
| conflict.stale_fact_exposure_rate | 0.5 | 0 | -0.5 ✅ |
| conflict.active_fact_accuracy | 0.5 | 1 | 0.5 ✅ |
| conflict.duplicate_fact_exposure_rate | 0.5 | 0 | -0.5 ✅ |
| **Lifecycle (retract / TTL)** | | | |
| lifecycle.retract_exposure_rate | 1 | 0 | -1 ✅ |
| lifecycle.expire_exposure_rate | 1 | 0 | -1 ✅ |
| **User Isolation** | | | |
| isolation.vector_leakage_rate | 0.5 | 0 | -0.5 ✅ |
| isolation.sql_leakage_rate | 1 | 0 | -1 ✅ |
| **Retrieval Quality** | | | |
| retrieval.recall_at_5 | 1 | 1 | 0 |
| retrieval.mrr | 1 | 1 | 0 |
| retrieval.fact_precision | 0.5 | 0.75 | 0.25 ✅ |
| **Continuation** | | | |
| continuation.required_fact_hit_rate | 1 | 1 | 0 |
| continuation.specific_value_retention_rate | 1 | 1 | 0 |

## 结论

1. **Improved 三大优势得到量化**：冲突消解（stale 曝光 0.5→0、active 准确率 0.5→1、
   重复曝光 0.5→0）、生命周期（retract/TTL 曝光 1→0）、用户隔离（vector 0.5→0、SQL 1→0）。
2. **recall@5 / MRR 两版都是 1.0**：hash embedding 非语义，相关 memory 的 token overlap
   最高、总排第 1；两版 embedding 完全相同，memory 级召回不区分（与 `baseline_analysis.md`
   预测一致）。MRR 的区分力需要真实 embedding + 更大语料才能体现。
3. **fact_precision 0.5 → 0.75** 是检索维度真正可量化的差异，精确来自两处：
   (a) supersede 过滤掉 stale 的 `current_status=unknown`；(b) user_id 隔离过滤掉 bob 泄漏的
   `999delX`。这两条在 baseline 里都进了候选，improved 都剔掉了。
4. **continuation 两版都是 1.0**：两版都逐字保留 specific value（185delAG/5382insC），
   正向保留本就不是差异点；stale 值消歧差异已由 conflict 指标覆盖。
5. **额外运行时发现（最重要）**：baseline 的 `get_facts_by_memory` / `retrieve()` 在当前依赖
   环境下**直接崩溃** —— `dataclasses.asdict()` 作用在 SQLAlchemy ORM 行上（TypeError），
   以及 `retrieve()` 把字符串 memory_id 传给 `Uuid` 列（AttributeError: 'str' has no 'hex'）。
   runner 通过直接读 DB（不改生产代码）绕开这两处崩溃以得到数值结果，但这一崩溃本身就是
   baseline 与 improved 之间最实质的差距：baseline 的事实检索路径不可用。
