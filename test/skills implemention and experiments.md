# Skills implemention and experiments

Main code changes:
- Added Skills metadata (name/description) defined under [`biomni/skills/`](../biomni/skills/), e.g. `SKILL.md` files used for Stage 1 retrieval.
- Two-stage retrieval routing implemented in [`biomni/agent/a1.py`](../biomni/agent/a1.py) (Stage 1 skill selection → Stage 2 tool selection).
- Retrieval token usage logging updated in [`biomni/model/retriever.py`](../biomni/model/retriever.py) (`ToolRetriever.prompt_based_retrieval`).

Experiments logs:
- [test/retrieval_baseline_single_stage.log.txt](./retrieval_baseline_single_stage.log.txt)
- [test/retrieval_two_stage_skill_tool.log.txt](./retrieval_two_stage_skill_tool.log.txt)

### Single-stage (baseline)
- **Tools provided to LLM**: 220
- **Tools selected**: 14
- **Libraries selected**: 7
- **Know-how selected**: 1
- **Data lake selected**: 0
- **Tokens**: Input 12,181 / Output 170 / Total 12,351

### Two-stage(skills)
- **Selected skills**: 6
- **Tools selected**: 10
- **Tools provided to LLM (Stage 2)**: 97
- **Libraries selected**: 8
- **Know-how selected**: 1
- **Data lake selected**: 0
- **Tokens**: Input 8,609 / Output 419 / Total 9,028

- **Input saved**: 12,181 − 8,609 = 3,572（about 29.3%）
- **Total saved**: 12,351 − 9,028 = 3,323（about 26.9%）
