# Retrieval Routing Token Comparison

Raw logs:
- `test/retrieval_baseline_single_stage.log.txt`
- `test/retrieval_two_stage_skill_tool.log.txt`

### Single-stage (baseline)
- **Tools provided to LLM**: 220
- **Tools selected**: 14
- **Libraries selected**: 7
- **Know-how selected**: 1
- **Data lake selected**: 0
- **Tokens**: Input 12,181 / Output 170 / Total 12,351

### Two-stage
- **Selected skills**: 6
- **Tools selected**: 10
- **Tools provided to LLM (Stage 2)**: 97
- **Libraries selected**: 8
- **Know-how selected**: 1
- **Data lake selected**: 0
- **Tokens**: Input 8,609 / Output 419 / Total 9,028

- **Input saved**: 12,181 − 8,609 = 3,572（about 29.3%）
- **Total saved**: 12,351 − 9,028 = 3,323（about 26.9%）