# Memory Benchmark — 最终结果（baseline → improved，固定版本）

版本：improved = 当前树 `<repo-root>`；baseline = 4349ab3 `<baseline-checkout>`。
venv：`<benchmark-venv>`（Python 3.14.4，chromadb 1.5.9 + ONNX all-MiniLM-L6-v2）。
本版本已固定，包含：user_id 隔离加固（Phase A）+ Query-aware Fact Reranking（0.7/0.3，无 hard threshold）。

三个 benchmark 组件，全部跑 baseline + improved：

---

## A. Differentiated benchmark（`run_benchmark.py`，hash embedding，覆盖存储/冲突/生命周期/隔离机制）

| metric | baseline | improved | delta |
|--------|---------:|---------:|------:|
| **Conflict Resolution** | | | |
| stale_fact_exposure_rate | 0.5 | 0 | -0.5 ✅ |
| active_fact_accuracy | 0.5 | 1 | 0.5 ✅ |
| duplicate_fact_exposure_rate | 0.5 | 0 | -0.5 ✅ |
| **Lifecycle (retract / TTL)** | | | |
| retract_exposure_rate | 1 | 0 | -1 ✅ |
| expire_exposure_rate | 1 | 0 | -1 ✅ |
| **User Isolation** | | | |
| vector_leakage_rate | 0.5 | 0 | -0.5 ✅ |
| sql_leakage_rate | 1 | 0 | -1 ✅ |
| **Retrieval Quality** | | | |
| recall_at_5 | 1 | 1 | 0 |
| mrr | 1 | 1 | 0 |
| fact_precision | 0.5 | 0.75 | 0.25 ✅ |
| **Continuation** | | | |
| required_fact_hit_rate | 1 | 1 | 0 |
| specific_value_retention_rate | 1 | 1 | 0 |

> recall/mrr/continuation 两版都是 1.0，因为 hash embedding 非语义（相关 memory token overlap 最高），
> memory 级召回无区分力；差异集中在 conflict/lifecycle/isolation/fact_precision。

## B. Retrieval case benchmark（`retrieval/run_retrieval.py`，真实 semantic embedding，20 用例 / 20 记忆，TOP_K=5）

| 指标 | baseline | improved | Δ |
| --- | --- | --- | --- |
| precision_at_1 | 0.5000 | **0.5500** | +0.0500 |
| precision_at_3 | 0.2333 | **0.2833** | +0.0500 |
| precision_at_5 | 0.1600 | **0.1900** | +0.0300 |
| recall_at_5 | 0.7250 | **0.8250** | +0.1000 |
| mrr | 0.5808 | **0.6683** | +0.0875 |
| fact_precision（n=3） | 0.4889 | **0.5000** | +0.0111 |

分类 MRR（baseline → improved）：exact_lexical 1.0→1.0；paraphrase 0.0667→**0.1778**；
low_lexical_overlap 0.4444→**0.5**；specific_fact 0.7778→**0.8333**；previous_task 0.6667→0.6667；
multi_memory 0.4167→**0.6111**；user_isolation 0.75→**1.0**（leak_rate 1.0→**0.0**）。

## C. Continuation case benchmark（`continuation/run_continuation.py`，真实 embedding，15 用例 / 10 类）

| 版本 | 通过 | 通过率 |
| --- | --- | --- |
| baseline | 0/15 | 0.00 |
| improved | **14/15** | **0.9333** |

唯一剩余失败：**cont_002**（`egfr_differential_expression` 在 corpus 里属于 user_alice，但用例
`session_a.user_id` 写成 user_bob）——基准数据不一致，非代码问题。

## 结论

1. **用户隔离从 0 到 1**：vector/sql leak 全部归零，user_isolation 类 leak_rate 1.0→0.0。
2. **冲突消解 + 生命周期**：stale/dup 曝光 0.5→0，active 准确率 0.5→1，retract/TTL 曝光 1→0。
3. **Memory 级召回全面提升**：R@5 0.725→0.825，MRR 0.5808→0.6683，多记忆召回 R@5 0.5→0.8333。
4. **Query-aware Fact Reranking 生效**：continuation 通过率 5/15 → 14/15（去掉 hard threshold 后
   关键事实不再被误删）；retrieval fact_precision 由回退 0.3333 修复为正增益 0.5。
