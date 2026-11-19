# Continuation Benchmark 结果（固定版本）

跨会话任务续接能力验证：session_a 摄入 → 生成记忆 → session_b 查询 → 检索 MemoryContext → 校验成功标准。
评判对象是**检索行为**（是否召回上次记忆、关键事实是否在上下文、是否知道停在哪里、是否避免重启、是否跨用户泄露），**不评判回答质量**。
本版本含 Query-aware Fact Reranking（0.7/0.3，无 hard threshold）。

- 语料：20 条记忆（11 user_alice + 9 user_bob）
- 用例：15 个（10 类）
- 真实 embedding：`all-MiniLM-L6-v2`（384-d，ONNX 后端）

## 总览

| 版本 | 通过 | 失败 | 通过率 |
| --- | --- | --- | --- |
| baseline（4349ab3，无 user_id 隔离） | 0 | 15 | 0.00 |
| improved（当前树） | 14 | 1 | 0.9333 |

## 分类结果（improved）

| 类别 | n | 通过 | 通过率 |
| --- | --- | --- | --- |
| user_isolation | 2 | 2 | 1.00 |
| reuse_selected_object | 1 | 1 | 1.00 |
| know_incomplete_work | 2 | 2 | 1.00 |
| know_completed_work | 1 | 1 | 1.00 |
| know_last_step | 1 | 1 | 1.00 |
| reuse_specific_conclusion | 1 | 1 | 1.00 |
| reuse_tool_result | 2 | 2 | 1.00 |
| cross_memory_continuation | 1 | 1 | 1.00 |
| continue_previous_analysis | 1 | 1 | 1.00 |
| reuse_specific_value | 3 | 2 | 0.67 |

## 结论

1. **用户隔离已修复**：improved 2/2（baseline 0/2 全线跨用户泄露）。
2. **记忆召回 + 关键事实召回都正常**：14/15 用例通过（记忆召回了、关键事实也进入上下文，且无泄露）。
3. 唯一剩余失败：**cont_002**，属基准数据不一致，非代码问题。

## 暴露的问题（仅记录，不修复）

### P1. 基准数据不一致（cont_002）
- **problem**：`continuation_cases.json` 中 cont_002 的 `session_a.user_id = "user_bob"`，但
  `memory_corpus.json` 中 `egfr_differential_expression.user_id = "user_alice"`。user_bob 查询时
  无法召回 user_alice 的 EGFR 记忆（隔离**正确**），导致 `recalled=0` 误判。
- **location**：`continuation/continuation_cases.json`（cont_002）与 `memory_corpus.json`（egfr_differential_expression）
- **impact**：cont_002 误判为召回失败，实际是基准数据把记忆挂错了用户。
