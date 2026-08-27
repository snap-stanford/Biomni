"""Prompt templates for memory extraction."""

MEMORY_EXTRACTION_PROMPT = """\
You are a memory extraction system for an autonomous biomedical agent.

Given a trace of a completed agent conversation, produce a structured object with:

1. `summary`: a concise, self-contained summary of what the task was and what the agent
   accomplished. It must contain enough detail that a future session could resume the
   work without seeing the original messages. Include key parameters, tools used, and
   results obtained.

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

Conversation trace:
{trace}

Return ONLY the structured object.
"""
