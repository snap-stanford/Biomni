# Biomni Progress Report — March 2026

## 1. Retrieval Architecture

- Designed and implemented a two-stage retrieval pipeline: Stage-1 selects relevant skill modules from 22 candidates; Stage-2 expands and retrieves tools only under the selected skills, reducing the candidate set size
- Achieved 26–29% input token reduction on benchmark query ("Query PubMed for papers about CRISPR and return 5 results"), from 12,181 to 8,609 input tokens
- Implemented split-model routing: retrieval stages run on Haiku, reasoning/code generation runs on Sonnet, reducing retrieval cost by ~75–80%
- Exposed retrieval model selection, two-stage toggle, and routing controls as runtime environment variables, enabling configuration changes without code modification

## 2. Evaluation & Debugging

- Built a Stage-1 skill retrieval evaluation script covering 10 biomedical queries with automated hit-rate calculation and per-case OK/MISS labeling
- Each evaluation case logs expected vs. selected skills for fast identification of retrieval drift
- Evaluation model is switchable via `--model` parameter, enabling head-to-head comparison of different models on the retrieval task

## 3. Caching & Cost Optimization

- Implemented a two-layer retrieval cache (in-memory LRU + disk persistence) that skips LLM calls entirely on cache hit
- Supports configurable TTL (default 24h), max entry count (default 2,000), and LRU eviction
- Introduced cache key versioning so prompt template changes automatically invalidate stale entries
- Each retrieval call emits structured logs with cache hit/miss status, latency, and selected tool count

## 4. Dependency Checks & Execution Robustness

- Implemented AST-level dependency scanning at startup to detect missing Python packages before any tool execution
- Provided three enforcement modes (off / warn / strict) with two check timings (startup, post-retrieval, or both)
- Built an allowlist-based auto-install flow that detects the runtime environment (conda / venv / system) and defaults to installing only within virtual environments
- Added dry-run mode to preview pending installations without executing them

## Quantitative Summary

| Metric | Value |
|---|---|
| Input token savings (two-stage vs. baseline) | 26–29% |
| Retrieval cost savings (Haiku vs. Sonnet) | ~75–80% |
| LLM calls on cache hit | 0 |
| Skill eval coverage | 10 cases |
