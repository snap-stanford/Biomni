# Retrieval Benchmark 结果（真实 embedding，固定版本）

Memory 级 Recall@K / MRR / Precision@K + Fact 级 fact_precision + 分类拆解，baseline vs improved。
真实 embedding：`sentence_transformer` = `all-MiniLM-L6-v2`（384-d，ONNX 后端）。
本版本含 Query-aware Fact Reranking（0.7/0.3，无 hard threshold）。

- 语料：20 条记忆（11 alice + 9 bob）；用例：20 个（7 类）；TOP_K=5

## Memory 级

| 指标 | baseline | improved | Δ |
| --- | --- | --- | --- |
| precision_at_1 | 0.5000 | **0.5500** | +0.0500 |
| precision_at_3 | 0.2333 | **0.2833** | +0.0500 |
| precision_at_5 | 0.1600 | **0.1900** | +0.0300 |
| recall_at_5 | 0.7250 | **0.8250** | +0.1000 |
| mrr | 0.5808 | **0.6683** | +0.0875 |

## Fact 级

| 指标 | baseline | improved | Δ |
| --- | --- | --- | --- |
| fact_precision（n=3 specific_fact） | 0.4889 | **0.5000** | +0.0111 |

## 分类拆解（improved vs baseline）

| 类别 | 指标 | baseline | improved |
| --- | --- | --- | --- |
| exact_lexical | P@1 / R@5 / MRR | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| paraphrase | P@3 / R@5 / MRR | 0.0 / 0.3333 / 0.0667 | **0.1111 / 0.6667 / 0.1778** |
| low_lexical_overlap | hit_rate / MRR | 0.6667 / 0.4444 | 0.6667 / **0.5000** |
| specific_fact | R@5 / MRR | 1.0 / 0.7778 | 1.0 / **0.8333** |
| previous_task | P@1 / R@5 / MRR | 0.6667 / 0.6667 / 0.6667 | 0.6667 / 0.6667 / 0.6667 |
| multi_memory | R@5 / MRR | 0.5 / 0.4167 | **0.8333 / 0.6111** |
| user_isolation | P@1 / leak_rate | 0.5 / **1.0** | **1.0 / 0.0** |

## 结论

1. **Memory 级全面提升**：R@5 0.725 → 0.825，MRR 0.5808 → 0.6683。
2. **多记忆召回显著改善**：multi_memory R@5 0.5 → 0.8333。
3. **用户隔离从 0 到 1**：user_isolation leak_rate 1.0 → 0.0，P@1 0.5 → 1.0。
4. **fact 级回正**：去掉 hard threshold 后 fact_precision 由（旧版回退的）0.3333 恢复为正增益 0.5。
