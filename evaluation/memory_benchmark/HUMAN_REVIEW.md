# HUMAN_REVIEW：Ground Truth 人工标注指南

> 本目录的所有 Ground Truth **只能由人工确认**，禁止把 LLM / Agent 自动生成的答案
> 直接当作 Ground Truth。Claude 可以**生成候选答案供参考**，但绝不能自动把它们写入
> `expected_*` 字段并标记为已确认。

---

## 0. 总原则

1. **谁写 Ground Truth**：人类标注者（可以是用户本人，或用户明确授权后人工核对）。
2. **候选答案**：可以由模型/Agent 预填到草稿字段（如 `candidate_*`），但必须人工逐条
   核对并移动到 `expected_*`，同时把 `review_status` 改为 `human_confirmed`。
3. **可判定**：所有断言必须能客观判定真假，禁止「看起来合理」「大致正确」这类模糊表述。
4. **不越界**：标注过程中不修改任何生产代码（`memory/*.py`、`database/*.py`、Agent 代码）。

---

## 1. Extraction 标注（extraction_cases.json）

每个 case 的 `trace` 已经是固定输入，需人工填写两个字段：

### 1.1 `expected_facts`（应提取出的事实）

每条是一个 (entity, relation, value, source) 四元组：

```json
{"entity": "BRCA1", "relation": "pathogenic_mutation", "value": "185delAG", "source": "tool_result"}
```

标注方法：
- 逐条读 trace 的 `observation` 与 `solution` 消息，提取「主语-谓语-宾语」三元组。
- `source` 按事实来源填：`tool_result`（工具返回的数值）、`user`（用户明确给出）、`llm`（模型推断）。
- **只写确定性事实**，不写推测；数值要精确（如 `0.75` 而不是「较高」）。
- 不确定是否该算 fact 时，先写进 `candidate_expected_facts`，人工定夺后再进 `expected_facts`。

### 1.2 `expected_summary_information`（摘要应覆盖的信息点）

每条是 (category, content)：

```json
{"category": "observation_results", "content": "EGFR log2FC=2.3, padj=0.001；共 412 个差异基因"}
```

`category` 与下面第 4 节「Summary Quality」6 维度的对应关系：

| category | 对应维度 |
|----------|----------|
| `task_context` | D1 任务与上下文 |
| `tools_params` | D2 工具与关键参数 |
| `observation_results` | D3 观察结果 |
| `solution_conclusion` | D4 结论 |
| `status_remaining` | D5 完成状态 + D6 可续接性 |

标注方法：对每个 case，列「一个合格 summary **必须**包含」的关键信息点（不是逐字复述，
而是「缺了它 summary 就扣分」的硬性要求）。信息点要具体到值。

---

## 2. Retrieval 标注（retrieval_cases.json + memory_corpus.json）

Ground Truth 已改为 **`expected_memory_keys`（稳定语义标识）**，不再是 UUID。

### 2.1 确认 `memory_corpus.json` 的 memory_key 归属

- 每条 memory 的 `memory_key` 是否语义准确、是否与 `trace_ref` 指向的 case 内容一致。
- 每条 memory 的 `user_id` 归属是否正确（这直接决定 user_isolation 测试是否成立）。

### 2.2 逐条确认 `expected_memory_keys` / `excluded_memory_keys`

- 对每个 retrieval case，判断「query 应该命中 corpus 中的哪些 memory」，把命中的
  `memory_key` 写入 `expected_memory_keys`。
- 对 `user_isolation` case，额外写 `excluded_memory_keys`（明确不应命中的他人 memory）。
- **不写 UUID**：UUID 是 ingest 时随机生成的，跨版本不同。真实 memory_id 由 runner 运行时
  通过 `memory_key -> memory_id` 映射表解析（见 README「memory_key 机制」）。

### 2.3 标注注意

- `low_lexical_overlap` 类 case：这些 query 与 memory 几乎无共享词，是**为了真实 embedding**
  设计的。在 hash embedding 下预期会失败——这是**有效发现**，不是 bug。标注时仍按「语义上
  应该命中」来标。
- `multi_memory` 类 case：`expected_memory_keys` 必须包含**同属一个 user** 的多条 memory。

---

## 3. Continuation 标注（continuation_cases.json）

### 3.1 确认 `memory_keys` 归属

- 每个 scenario 的 `session_a.memory_keys` 应指向 corpus 中正确的 memory_key（session_a
  的任务即 ingest 该 memory 的 trace）。
- `cross_memory_continuation`（cont_012）应指向 ≥2 条同 user 的 memory。

### 3.2 确认 `expected_memory`（session_a 应沉淀的 memory 内容）

- `summary_must_contain`：列出 session_a 的 summary **必须包含**的具体值/结论。
- `facts_must_contain`：列出应沉淀的 fact 三元组。

### 3.3 确认 `success_criteria` 可判定

每条 success_criteria 必须满足「换一个人来判，结论一致」。检查清单：
- ❌ 「回答正确」→ ✅ 「上下文明确出现 185delAG 和 5382insC 两个具体值」
- ❌ 「没有遗忘」→ ✅ 「session_b 不再调用 fastqc，直接进入比对」
- ❌ 「隔离良好」→ ✅ 「session_b 结果不含 user_alice 的 memory/fact」

---

## 4. Summary Quality 评分标准（6 维度，0–12 分）

> 用于 **Extraction Benchmark** 中给「模型产出的 summary」打分。每个维度 0/1/2 分，
> 六维相加得 0–12 分。评分对象是**模型实际产出的 summary**，对照的是
> `expected_summary_information`（Ground Truth）。

| # | 维度 | 0 分（缺失） | 1 分（部分） | 2 分（完整且有用） |
|---|------|--------------|--------------|---------------------|
| **D1** | **任务与上下文** Task/Context | 未说明本次在做什么 | 说了任务方向但缺上下文/对象 | 清楚说明任务目标、对象（基因/样本/药物）与背景 |
| **D2** | **工具与关键参数** Tools/Key Parameters | 未提及用了什么工具/方法 | 提了工具但缺关键参数或输入 | 工具 + 关键参数（阈值、输入文件、方法名）齐备 |
| **D3** | **观察结果** Observation | 未记录任何实际结果 | 只给结论、缺具体观察数值 | 记录了工具返回的具体数值/观察（如 log2FC=2.3） |
| **D4** | **结论** Solution | 观察与结论混淆或缺失 | 有结论但依据不清 | 明确区分观察与结论，给出最终判定及依据 |
| **D5** | **完成状态** Task Status | 未说明是否完成/结果状态 | 状态含糊 | 明确说明任务完成度、结果是否合格/通过 |
| **D6** | **可续接性** Future Usefulness | 无法据此继续后续工作 | 部分信息可复用但缺关键值 | 含足量具体值与未完成项，未来可无缝续接 |

### 4.1 打分流程

1. 取一条模型产出的 summary。
2. 对照该 case 的 `expected_summary_information`，逐维度判断 0/1/2。
3. 六维相加得总分（0–12）。
4. 用一句「扣分理由」记录缺了什么（便于回溯），不要只写分数。

### 4.2 打分示例

**Ground Truth（ext_002）期望包含**：EGFR、log2FC=2.3、padj=0.001、412 个差异基因、
「显著上调」结论。

| summary 内容 | D1 | D2 | D3 | D4 | D5 | D6 | 总分 | 理由 |
|---|---|---|---|---|---|---|---|---|
| "分析了 EGFR 在肿瘤 vs 正常中的差异表达（DESeq2，padj=0.05）。EGFR log2FC=2.3, padj=0.001；共 412 个差异基因。结论：EGFR 显著上调。" | 2 | 2 | 2 | 2 | 2 | 2 | **12** | 六维齐全且数值精确 |
| "做了差异表达分析，EGFR 上调了。" | 1 | 0 | 1 | 1 | 0 | 0 | **3** | 缺工具/参数、缺具体数值、无可续接信息 |
| "EGFR 上调 2.3 倍，差异显著。" | 1 | 0 | 2 | 2 | 0 | 1 | **6** | 有观察与结论，但缺工具、缺状态、缺规模信息 |

### 4.3 与 Fact Accuracy 的关系

- **Fact Accuracy** 看 `expected_facts` 是否被精确提取（Precision/Recall of facts）。
- **Summary Quality** 看 summary 的信息覆盖与可续接性（6 维度 0–12）。
- 两者独立打分，最终报告分别汇报。

---

## 5. 标注完成后

1. 把每个 case 的 `review_status` 改为 `human_confirmed`。
2. 把确认后的 Ground Truth 归档到 `ground_truth/`（可复制一份确认版）。
3. 运行阶段由 runner 使用 `ground_truth/` 下的确认版，`results/` 只写跑分输出。
