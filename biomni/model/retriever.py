###########################################################################
# Updated by Kyle:
# - Enhanced retrieval logging in ToolRetriever.prompt_based_retrieval:
#   - Added robust token usage extraction/printing (around Lines 100–170)
#   - Helps compare single-stage vs two-stage (skills-based) retrieval runs
###########################################################################

import contextlib

# Updated by Kyle
import copy
import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


class ToolRetriever:
    """Retrieve tools from the tool registry."""

    def __init__(self):
        # Updated by Kyle
        # Retrieval cache (in-memory + optional file persistence).
        self._retrieval_cache: OrderedDict[str, dict] = OrderedDict()
        self._retriever_prompt_version = "retriever_prompt_v1"
        self._cache_persist_enabled = self._env_bool("BIOMNI_RETRIEVAL_CACHE_PERSIST_ENABLED", True)
        self._cache_file_path = self._resolve_cache_file_path()
        self._load_cache_from_disk()
        if self._env_bool("BIOMNI_RETRIEVAL_CACHE_DEBUG", False):
            print(
                f"[RETRIEVAL CACHE INIT] persist={int(self._cache_persist_enabled)} "
                f"path={self._cache_file_path} loaded_entries={len(self._retrieval_cache)}"
            )

    # Updated by Kyle
    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    # Updated by Kyle
    @staticmethod
    def _env_int(name: str, default: int, min_value: int = 1) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        with contextlib.suppress(ValueError):
            return max(int(raw), min_value)
        return default

    # Updated by Kyle
    @staticmethod
    def _resolve_cache_file_path() -> str:
        raw = os.getenv("BIOMNI_RETRIEVAL_CACHE_PATH", "").strip()
        if raw:
            return str(Path(raw).expanduser().resolve())
        # Default to repo-root/data so path is stable across different launch CWDs.
        return str((Path(__file__).resolve().parents[2] / "data" / "retrieval_cache.json").resolve())

    # Updated by Kyle
    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(str(query).strip().lower().split())

    # Updated by Kyle
    @staticmethod
    def _resource_item_for_hash(item) -> dict:
        if isinstance(item, dict):
            return {
                "name": str(item.get("name", "")),
                "module": str(item.get("module", "")),
                "description": str(item.get("description", "")),
            }
        return {"name": str(item), "module": "", "description": ""}

    # Updated by Kyle
    def _resources_hash(self, resources: dict) -> str:
        def sorted_items(items):
            normalized = [self._resource_item_for_hash(x) for x in items]
            normalized.sort(key=lambda it: (it.get("name", ""), it.get("module", ""), it.get("description", "")))
            return normalized

        payload = {
            # Use order-insensitive hashing so semantically identical candidate sets hit cache.
            "tools": sorted_items(resources.get("tools", [])),
            "data_lake": sorted_items(resources.get("data_lake", [])),
            "libraries": sorted_items(resources.get("libraries", [])),
            "know_how": sorted_items(resources.get("know_how", [])),
        }
        stable = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()

    # Updated by Kyle
    @staticmethod
    def _llm_model_id(llm) -> str:
        for attr in ("model_name", "model", "model_id"):
            value = getattr(llm, attr, None)
            if value:
                return str(value)
        return str(type(llm))

    # Updated by Kyle
    def _cache_key(self, stage: str, model_id: str, query: str, resources: dict) -> str:
        key_payload = {
            "stage": stage,
            "model_id": model_id,
            "query_norm": self._normalize_query(query),
            "resources_hash": self._resources_hash(resources),
            "retriever_prompt_version": self._retriever_prompt_version,
        }
        stable = json.dumps(key_payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()

    # Updated by Kyle
    def _cache_get(self, key: str) -> dict | None:
        now = time.time()
        changed = False
        entry = self._retrieval_cache.get(key)
        if not entry:
            return None
        if now >= entry.get("expires_at", 0):
            self._retrieval_cache.pop(key, None)
            changed = True
            if changed:
                self._persist_cache_to_disk()
            return None
        self._retrieval_cache.move_to_end(key)
        return entry

    # Updated by Kyle
    def _cache_set(
        self, key: str, selected_resources: dict, ttl_seconds: int, max_entries: int, latency_ms: float
    ) -> None:
        now = time.time()
        self._retrieval_cache[key] = {
            "selected_resources": copy.deepcopy(selected_resources),
            "meta": {
                "created_at": now,
                "latency_ms": latency_ms,
                "cache_version": self._retriever_prompt_version,
            },
            "expires_at": now + ttl_seconds,
        }
        self._retrieval_cache.move_to_end(key)
        self._prune_cache(max_entries=max_entries, now=now)
        self._persist_cache_to_disk()

    # Updated by Kyle
    def _prune_cache(self, max_entries: int, now: float | None = None) -> None:
        now = now if now is not None else time.time()

        # Drop expired entries first.
        expired_keys = [k for k, v in self._retrieval_cache.items() if now >= v.get("expires_at", 0)]
        for key in expired_keys:
            self._retrieval_cache.pop(key, None)

        # Enforce LRU size limit.
        while len(self._retrieval_cache) > max_entries:
            self._retrieval_cache.popitem(last=False)

    # Updated by Kyle
    def _load_cache_from_disk(self) -> None:
        if not self._cache_persist_enabled:
            return

        path = Path(self._cache_file_path)
        if not path.exists():
            return

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        entries = raw.get("entries", [])
        loaded: OrderedDict[str, dict] = OrderedDict()
        if isinstance(entries, list):
            for item in entries:
                if not isinstance(item, dict):
                    continue
                key = item.get("key")
                value = item.get("value")
                if isinstance(key, str) and isinstance(value, dict):
                    loaded[key] = value

        self._retrieval_cache = loaded
        self._prune_cache(max_entries=self._env_int("BIOMNI_RETRIEVAL_CACHE_MAX_ENTRIES", 2000, min_value=1))

    # Updated by Kyle
    def _persist_cache_to_disk(self) -> None:
        if not self._cache_persist_enabled:
            return

        path = Path(self._cache_file_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "entries": [{"key": k, "value": v} for k, v in self._retrieval_cache.items()],
            }
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp_path, path)
        except Exception:
            # Best-effort persistence; retrieval should continue even if disk cache fails.
            return

    # Updated by Kyle
    def _log_cache_line(
        self, *, stage: str, model_id: str, cache_hit: int, latency_ms: float, selected_resources: dict
    ) -> None:
        print(
            "[RETRIEVAL] "
            f"stage={stage} model={model_id} cache_hit={cache_hit} "
            f"latency_ms={latency_ms:.1f} selected_tools_count={len(selected_resources.get('tools', []))}"
        )

    # Updated by Kyle
    @staticmethod
    def _suspected_parse_failure(response_content, selected_indices: dict, resources: dict) -> bool:
        if not resources.get("tools") or selected_indices.get("tools"):
            return False

        text = response_content if isinstance(response_content, str) else str(response_content)
        normalized = text.replace("**", "").replace("__", "")
        tools_match = re.search(r"TOOLS\s*:\s*\[(.*?)\]", normalized, re.IGNORECASE | re.DOTALL)
        if tools_match:
            return bool(tools_match.group(1).strip())
        return "TOOLS" in normalized.upper()

    # Updated by Kyle
    def _response_to_text(self, response_content) -> str:
        """Normalize heterogeneous provider responses into plain text for parsing/logging."""
        if isinstance(response_content, str):
            return response_content
        if response_content is None:
            return ""

        if isinstance(response_content, dict):
            if isinstance(response_content.get("text"), str):
                return response_content["text"]
            if "content" in response_content:
                return self._response_to_text(response_content.get("content"))
            if "delta" in response_content:
                return self._response_to_text(response_content.get("delta"))
            text_values = [v for v in response_content.values() if isinstance(v, str)]
            if text_values:
                return "\n".join(text_values)
            return str(response_content)

        if isinstance(response_content, list):
            parts = []
            for item in response_content:
                part = self._response_to_text(item)
                if part:
                    parts.append(part)
            return "\n".join(parts)

        text_attr = getattr(response_content, "text", None)
        if isinstance(text_attr, str):
            return text_attr
        content_attr = getattr(response_content, "content", None)
        if content_attr is not None:
            return self._response_to_text(content_attr)
        return str(response_content)

    def prompt_based_retrieval(self, query: str, resources: dict, llm=None, stage: str = "retrieval") -> dict:
        """Use a prompt-based approach to retrieve the most relevant resources for a query.

        Args:
            query: The user's query
            resources: A dictionary with keys 'tools', 'data_lake', 'libraries', and 'know_how',
                      each containing a list of available resources
            llm: Optional LLM instance to use for retrieval (if None, will create a new one)
            stage: Retrieval stage label for logging/caching (e.g., skill_retrieval/tool_retrieval)

        Returns:
            A dictionary with the same keys, but containing only the most relevant resources

        """
        retrieval_start = time.perf_counter()

        # Use the provided LLM or create a new one
        if llm is None:
            llm = ChatOpenAI(model="gpt-4o")

        # Updated by Kyle
        # Retrieval cache lookup
        cache_enabled = self._env_bool("BIOMNI_RETRIEVAL_CACHE_ENABLED", True)
        cache_debug = self._env_bool("BIOMNI_RETRIEVAL_CACHE_DEBUG", False)
        cache_ttl = self._env_int("BIOMNI_RETRIEVAL_CACHE_TTL_SECONDS", 86400, min_value=1)
        cache_max_entries = self._env_int("BIOMNI_RETRIEVAL_CACHE_MAX_ENTRIES", 2000, min_value=1)
        model_id = self._llm_model_id(llm)
        cache_key = self._cache_key(stage=stage, model_id=model_id, query=query, resources=resources)

        if cache_enabled:
            cached_entry = self._cache_get(cache_key)
            if cached_entry:
                latency_ms = (time.perf_counter() - retrieval_start) * 1000.0
                cached_resources = copy.deepcopy(cached_entry["selected_resources"])
                print(f"[RETRIEVAL CACHE HIT] stage={stage} model={model_id}")
                if cache_debug:
                    print(f"[RETRIEVAL CACHE DEBUG] key={cache_key} entries={len(self._retrieval_cache)}")
                self._log_cache_line(
                    stage=stage,
                    model_id=model_id,
                    cache_hit=1,
                    latency_ms=latency_ms,
                    selected_resources=cached_resources,
                )
                return cached_resources

        # Build prompt sections for available resources
        prompt_sections = []
        prompt_sections.append(f"""
You are an expert biomedical research assistant. Your task is to select the relevant resources to help answer a user's query.

USER QUERY: {query}

Below are the available resources. For each category, select items that are directly or indirectly relevant to answering the query.
Be generous in your selection - include resources that might be useful for the task, even if they're not explicitly mentioned in the query.
It's better to include slightly more resources than to miss potentially useful ones.

AVAILABLE TOOLS:
{self._format_resources_for_prompt(resources.get("tools", []))}

AVAILABLE DATA LAKE ITEMS:
{self._format_resources_for_prompt(resources.get("data_lake", []))}

AVAILABLE SOFTWARE LIBRARIES:
{self._format_resources_for_prompt(resources.get("libraries", []))}""")

        # Add know-how section if available
        if "know_how" in resources and resources["know_how"]:
            prompt_sections.append(f"""
AVAILABLE KNOW-HOW DOCUMENTS (Best Practices & Protocols):
{self._format_resources_for_prompt(resources.get("know_how", []))}""")

        # Build response format based on available categories
        response_format = """
For each category, respond with ONLY the indices of the relevant items in the following format:
TOOLS: [list of indices]
DATA_LAKE: [list of indices]
LIBRARIES: [list of indices]"""

        if "know_how" in resources and resources["know_how"]:
            response_format += "\nKNOW_HOW: [list of indices]"

        response_format += """

For example:
TOOLS: [0, 3, 5, 7, 9]
DATA_LAKE: [1, 2, 4]
LIBRARIES: [0, 2, 4, 5, 8]"""

        if "know_how" in resources and resources["know_how"]:
            response_format += "\nKNOW_HOW: [0, 1]"

        response_format += """

If a category has no relevant items, use an empty list, e.g., DATA_LAKE: []

IMPORTANT GUIDELINES:
1. Be generous but not excessive - aim to include all potentially relevant resources
2. ALWAYS prioritize database tools for general queries - include as many database tools as possible
3. Include all literature search tools
4. For wet lab sequence type of queries, ALWAYS include molecular biology tools
5. For data lake items, include datasets that could provide useful information
6. For libraries, include those that provide functions needed for analysis
7. For know-how documents, include those that provide relevant protocols, best practices, or troubleshooting guidance
8. Don't exclude resources just because they're not explicitly mentioned in the query
9. When in doubt about a database tool or molecular biology tool, include it rather than exclude it
"""

        prompt = "\n".join(prompt_sections) + response_format

        # Invoke the LLM
        if hasattr(llm, "invoke"):
            # For LangChain-style LLMs
            response = llm.invoke([HumanMessage(content=prompt)])
            response_content = response.content

            # Extract token usage -- Kyle
            token_usage = {}

            # Try to extract token usage from response_metadata
            if hasattr(response, "response_metadata") and response.response_metadata:
                metadata = response.response_metadata

                if isinstance(metadata, dict):
                    input_tokens = (
                        metadata.get("input_tokens")
                        or metadata.get("prompt_tokens")
                        or (metadata.get("usage", {}) if isinstance(metadata.get("usage"), dict) else {}).get(
                            "input_tokens"
                        )
                        or (metadata.get("usage", {}) if isinstance(metadata.get("usage"), dict) else {}).get(
                            "prompt_tokens"
                        )
                        or 0
                    )
                    output_tokens = (
                        metadata.get("output_tokens")
                        or metadata.get("completion_tokens")
                        or (metadata.get("usage", {}) if isinstance(metadata.get("usage"), dict) else {}).get(
                            "output_tokens"
                        )
                        or (metadata.get("usage", {}) if isinstance(metadata.get("usage"), dict) else {}).get(
                            "completion_tokens"
                        )
                        or 0
                    )

                    if input_tokens > 0 or output_tokens > 0:
                        token_usage = {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "cache_creation_input_tokens": metadata.get("cache_creation_input_tokens", 0),
                            "cache_read_input_tokens": metadata.get("cache_read_input_tokens", 0),
                        }

            # Fallback: try usage_metadata
            if not token_usage and hasattr(response, "usage_metadata") and response.usage_metadata:
                usage_meta = response.usage_metadata
                input_tokens = (
                    getattr(usage_meta, "input_tokens", None) or getattr(usage_meta, "prompt_tokens", None) or 0
                )
                output_tokens = (
                    getattr(usage_meta, "output_tokens", None) or getattr(usage_meta, "completion_tokens", None) or 0
                )

                if input_tokens > 0 or output_tokens > 0:
                    token_usage = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }

            # Print token usage summary
            if token_usage and (token_usage.get("input_tokens", 0) > 0 or token_usage.get("output_tokens", 0) > 0):
                total = token_usage.get("input_tokens", 0) + token_usage.get("output_tokens", 0)
                print("\n" + "=" * 60)
                print("📊 RETRIEVER TOKEN USAGE")
                print("=" * 60)
                print(f"  Input tokens:  {token_usage.get('input_tokens', 0)}")
                print(f"  Output tokens: {token_usage.get('output_tokens', 0)}")
                if token_usage.get("cache_creation_input_tokens"):
                    print(f"  Cache creation tokens: {token_usage.get('cache_creation_input_tokens')}")
                if token_usage.get("cache_read_input_tokens"):
                    print(f"  Cache read tokens: {token_usage.get('cache_read_input_tokens')}")
                print(f"  Total tokens:  {total}")
                print("=" * 60 + "\n")
        else:
            # For other LLM interfaces
            response_content = str(llm(prompt))

        # Updated by Kyle
        normalized_response_text = self._response_to_text(response_content)

        # Parse the response to extract the selected indices
        selected_indices = self._parse_llm_response(normalized_response_text)

        # Updated by Kyle
        # Debug assist: help diagnose parser-vs-model misses.
        debug_retriever = os.getenv("BIOMNI_RETRIEVER_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
        if debug_retriever and not selected_indices.get("tools") and resources.get("tools"):
            snippet = normalized_response_text
            snippet = snippet[:1200].replace("\n", "\\n")
            print("\n" + "=" * 60)
            print("⚠️ RETRIEVER DEBUG: parsed TOOLS is empty")
            print("Raw LLM response snippet:")
            print(snippet)
            print("=" * 60 + "\n")

        # Get the selected resources
        selected_resources = {
            "tools": [
                resources["tools"][i] for i in selected_indices.get("tools", []) if i < len(resources.get("tools", []))
            ],
            "data_lake": [
                resources["data_lake"][i]
                for i in selected_indices.get("data_lake", [])
                if i < len(resources.get("data_lake", []))
            ],
            "libraries": [
                resources["libraries"][i]
                for i in selected_indices.get("libraries", [])
                if i < len(resources.get("libraries", []))
            ],
        }

        # Add know-how if present
        if "know_how" in resources and resources["know_how"]:
            selected_resources["know_how"] = [
                resources["know_how"][i]
                for i in selected_indices.get("know_how", [])
                if i < len(resources.get("know_how", []))
            ]

        # Updated by Kyle
        # Cache write on miss (skip suspected parser-failure empties).
        latency_ms = (time.perf_counter() - retrieval_start) * 1000.0
        parse_failure_suspected = self._suspected_parse_failure(normalized_response_text, selected_indices, resources)
        if cache_enabled and not parse_failure_suspected:
            self._cache_set(
                key=cache_key,
                selected_resources=selected_resources,
                ttl_seconds=cache_ttl,
                max_entries=cache_max_entries,
                latency_ms=latency_ms,
            )
            print(f"[RETRIEVAL CACHE MISS -> STORE] stage={stage} model={model_id}")
            if cache_debug:
                print(f"[RETRIEVAL CACHE DEBUG] key={cache_key} entries={len(self._retrieval_cache)}")
        elif cache_enabled and parse_failure_suspected:
            print(f"[RETRIEVAL CACHE SKIP] stage={stage} reason=suspected_parse_failure")

        self._log_cache_line(
            stage=stage,
            model_id=model_id,
            cache_hit=0,
            latency_ms=latency_ms,
            selected_resources=selected_resources,
        )
        return selected_resources

    def _format_resources_for_prompt(self, resources: list) -> str:
        """Format resources for inclusion in the prompt."""
        formatted = []
        for i, resource in enumerate(resources):
            if isinstance(resource, dict):
                # Handle dictionary format (from tool registry or data lake/libraries with descriptions)
                name = resource.get("name", f"Resource {i}")
                description = resource.get("description", "")
                formatted.append(f"{i}. {name}: {description}")
            elif isinstance(resource, str):
                # Handle string format (simple strings)
                formatted.append(f"{i}. {resource}")
            else:
                # Try to extract name and description from tool objects
                name = getattr(resource, "name", str(resource))
                desc = getattr(resource, "description", "")
                formatted.append(f"{i}. {name}: {desc}")

        return "\n".join(formatted) if formatted else "None available"

    def _parse_llm_response(self, response) -> dict:
        """Parse the LLM response to extract the selected indices.

        Accepts either a plain string or a Responses API-style list of content blocks.
        """
        # Normalize response to string if it's a list of content blocks (Responses API)
        if isinstance(response, list):
            parts = []
            for item in response:
                # LangChain Responses API returns list of dicts like {"type": "text", "text": "..."}
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        parts.append(str(item.get("text", "")))
                    # If it's a tool_call or other block, ignore for this simple parsing
                elif isinstance(item, str):
                    parts.append(item)
            response = "\n".join([p for p in parts if p])
        elif not isinstance(response, str):
            response = str(response)
        # Updated by Kyle
        # Normalize simple markdown emphasis to improve downstream regex parsing.
        response = response.replace("**", "").replace("__", "")
        selected_indices = {"tools": [], "data_lake": [], "libraries": [], "know_how": []}

        # Updated by Kyle
        # Accept common markdown and label variants, e.g.:
        # TOOLS: [1,2], **TOOLS:** [1,2], DATA-LAKE: [...], KNOW HOW: [...]
        def extract_indices(label_pattern: str) -> list[int]:
            patterns = [
                rf"\*{{0,2}}\s*{label_pattern}\s*\*{{0,2}}\s*:\s*\[(.*?)\]",
                rf"{label_pattern}\s*:\s*\*{{0,2}}\s*\[(.*?)\]",
                rf"{label_pattern}\s*:\s*\[(.*?)\]",
            ]
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match and match.group(1).strip():
                    with contextlib.suppress(ValueError):
                        return [int(idx.strip()) for idx in match.group(1).split(",") if idx.strip()]
            return []

        selected_indices["tools"] = extract_indices(r"TOOLS")
        selected_indices["data_lake"] = extract_indices(r"DATA[\s_-]*LAKE")
        selected_indices["libraries"] = extract_indices(r"LIBRARIES")
        selected_indices["know_how"] = extract_indices(r"KNOW[\s_-]*HOW")

        return selected_indices
