"""Prompt templates for memory extraction."""

MEMORY_EXTRACTION_PROMPT = """\
You are a memory extraction system for an autonomous biomedical agent.

Given a trace of a completed agent conversation, produce a structured object with:

1. `summary`: a high-density, self-contained task summary that lets a future agent
   resume the work WITHOUT reading the original trace. It must capture, in this order:

   (a) Task / Context:
       - The user's goal for this task.
       - Key objects involved: datasets, genes, proteins, files, etc.

   (b) Tools / Key Parameters:
       - The important tools actually called, with their key input parameters.
       - Important configuration, thresholds, query conditions, file paths, etc.
       - Do NOT reproduce tool calls verbatim — keep only what has future value.

   (c) `<observation>` results:
       - The concrete results that tool/code execution produced: important numbers,
         experimental/analysis results, generated files, key findings.
       - Always record the specific result — never just "executed successfully".
       - Do NOT copy the observation verbatim.

   (d) `<solution>` conclusion:
       - The final answer, conclusion, or solution the agent arrived at.
       - Distinguish clearly: observation = what execution actually produced;
         solution = the conclusion the agent drew from those results.
       - If the task failed, record the failure reason.

   (e) Task Status / Remaining Work:
       - Whether the task is complete, partially complete, or failed.
       - If work remains, state what the next step should be.

   Goal: from the summary alone, a future agent should understand: what the user
   wanted -> which tools and key parameters were used -> what actual results were
   obtained -> what final conclusion was reached -> how far the task got and what
   comes next.

   Principles:
       - Prioritize information density over length.
       - Do not mechanically restate the trace.
       - Drop meaningless chatter and repeated information.
       - Do not keep long intermediate reasoning.
       - Keep key entities, parameters, numbers, results, and conclusions.
       - Merge multiple observations that describe the same result.
       - Omit tool calls that produced nothing useful for future tasks.
       - A failed task must still produce a useful summary.
       - The summary should be suitable for vector DB embedding and semantic retrieval.

2. `facts`: a list of durable, long-lived statements worth remembering beyond this session.

Rules for facts:
- Keep only long-term useful facts. Examples of what to KEEP:
  * user preferences and constraints
  * experimental results and derived conclusions
  * key parameters and identifiers (gene names, mutations, accession numbers)
  * task state worth resuming
  * concrete tool results
- Do NOT save low-value chatter ("hello", "thanks") or transient reasoning.
- Each fact must be subject-predicate-object: entity, relation, value.
- Set `confidence` between 0 and 1 reflecting how certain the trace makes the fact.
- Set `source` to exactly one of:
  * "tool_result" — the fact came from an executed tool's output
  * "user" — the fact was stated by the user
  * "llm" — the fact is inferred/guessed by the model (will be filtered downstream)
- Do NOT set `created_at`, `updated_at`, `status`, or `importance_score` — the
  system assigns these itself. Only output entity, relation, value, confidence,
  and source.

Conversation trace:
{trace}

Return ONLY the structured object.
"""
