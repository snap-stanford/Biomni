import contextlib
import re
import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings.bedrock import BedrockEmbeddings

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class ToolRetriever:
    """Retrieve tools from the tool registry."""

    def __init__(self):
        pass

    def prompt_based_retrieval(self, query: str, resources: dict, llm=None) -> dict:
        """Use a prompt-based approach to retrieve the most relevant resources for a query.

        Args:
            query: The user's query
            resources: A dictionary with keys 'tools', 'data_lake', 'libraries', and 'know_how',
                      each containing a list of available resources
            llm: Optional LLM instance to use for retrieval (if None, will create a new one)

        Returns:
            A dictionary with the same keys, but containing only the most relevant resources

        """
        # Build prompt sections for available resources
        prompt_sections = []
        prompt_sections.append(
            f"""
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
{self._format_resources_for_prompt(resources.get("libraries", []))}"""
        )

        # Add know-how section if available
        if "know_how" in resources and resources["know_how"]:
            know_how_formatted = self._format_resources_for_prompt(resources.get("know_how", []))
            prompt_sections.append(
                f"""
AVAILABLE KNOW-HOW DOCUMENTS (Best Practices & Protocols):
{know_how_formatted}"""
            )
            print(f"\n📚 KNOW-HOW IN RETRIEVAL PROMPT: {len(resources['know_how'])} documents available")
            for i, doc in enumerate(resources["know_how"]):
                if isinstance(doc, dict):
                    print(f"  [{i}] {doc.get('name', 'Unknown')} - {doc.get('description', 'No description')[:100]}")

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

        # Use the provided LLM or create a new one
        if llm is None:
            llm = ChatOpenAI(model="gpt-4o")

        # Invoke the LLM
        if hasattr(llm, "invoke"):
            # For LangChain-style LLMs
            response = llm.invoke([HumanMessage(content=prompt)])
            response_content = response.content
        else:
            # For other LLM interfaces
            response_content = str(llm(prompt))

        # Parse the response to extract the selected indices
        print(f"\n📚 LLM RETRIEVAL RESPONSE (first 500 chars): {response_content[:500]}")
        selected_indices = self._parse_llm_response(response_content)

        # Get the selected resources
        selected_resources = {
            "tools": [
                resources["tools"][i]
                for i in selected_indices.get("tools", [])
                if i < len(resources.get("tools", []))
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
            know_how_indices = selected_indices.get("know_how", [])
            selected_resources["know_how"] = [
                resources["know_how"][i]
                for i in know_how_indices
                if i < len(resources.get("know_how", []))
            ]
            # Log know-how selection
            if know_how_indices:
                print(f"\n📚 KNOW-HOW SELECTION: LLM selected indices {know_how_indices} from {len(resources['know_how'])} available documents")
            else:
                print(f"\n📚 KNOW-HOW SELECTION: LLM selected NO documents (empty list) from {len(resources['know_how'])} available documents")
        else:
            selected_resources["know_how"] = []
            if "know_how" not in resources:
                print(f"\n📚 KNOW-HOW SELECTION: 'know_how' key NOT FOUND in resources")
            elif not resources["know_how"]:
                print(f"\n📚 KNOW-HOW SELECTION: 'know_how' key exists but is EMPTY in resources")
            else:
                print(f"\n📚 KNOW-HOW SELECTION: No know-how documents available in resources")

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
        selected_indices = {
            "tools": [],
            "data_lake": [],
            "libraries": [],
            "know_how": [],
        }

        # Extract indices for each category
        tools_match = re.search(r"TOOLS:\s*\[(.*?)\]", response, re.IGNORECASE)
        if tools_match and tools_match.group(1).strip():
            with contextlib.suppress(ValueError):
                selected_indices["tools"] = [
                    int(idx.strip())
                    for idx in tools_match.group(1).split(",")
                    if idx.strip()
                ]

        data_lake_match = re.search(r"DATA_LAKE:\s*\[(.*?)\]", response, re.IGNORECASE)
        if data_lake_match and data_lake_match.group(1).strip():
            with contextlib.suppress(ValueError):
                selected_indices["data_lake"] = [
                    int(idx.strip())
                    for idx in data_lake_match.group(1).split(",")
                    if idx.strip()
                ]

        libraries_match = re.search(r"LIBRARIES:\s*\[(.*?)\]", response, re.IGNORECASE)
        if libraries_match and libraries_match.group(1).strip():
            with contextlib.suppress(ValueError):
                selected_indices["libraries"] = [
                    int(idx.strip())
                    for idx in libraries_match.group(1).split(",")
                    if idx.strip()
                ]

        # Extract know-how indices
        know_how_match = re.search(r"KNOW[-_]HOW:\s*\[(.*?)\]", response, re.IGNORECASE)
        if know_how_match:
            know_how_content = know_how_match.group(1).strip()
            if know_how_content:
                with contextlib.suppress(ValueError):
                    selected_indices["know_how"] = [
                        int(idx.strip())
                        for idx in know_how_content.split(",")
                        if idx.strip()
                    ]
            else:
                print(f"📚 KNOW-HOW PARSING: KNOW_HOW section found but empty (no indices selected)")
                selected_indices["know_how"] = []
        else:
            print(f"📚 KNOW-HOW PARSING: No KNOW_HOW section found in LLM response")
            # Check if know-how was mentioned in the response at all
            if "KNOW" in response.upper() or "HOW" in response.upper():
                print(f"📚 KNOW-HOW PARSING: Response contains 'KNOW' or 'HOW' but no valid KNOW_HOW: [...] format")

        return selected_indices


class ToolRetrieverByRAG:
    """Retrieve tools from the RAG database."""

    def __init__(self):
        pass

    def prompt_based_retrieval(self, query: str) -> dict:
        """Use a prompt-based approach to retrieve the most relevant resources for a query.

        Args:
            query: The user's query
            resources: A dictionary with keys 'tools', 'data_lake', and 'libraries',
                      each containing a list of available resources
            llm: Optional LLM instance to use for retrieval (if None, will create a new one)

        Returns:
            A dictionary with the same keys, but containing only the most relevant resources

        """
        # Create a prompt for the LLM to select relevant resources
        rag_query = f"""
You are an expert biomedical research assistant. Your task is to select the relevant resources to help answer a user's query.

IMPORTANT:
-----------------------------------
USER QUERY: {query}
-----------------------------------
Be generous in your selection - include resources that might be useful for the task, even if they're not explicitly mentioned in the query.
It's better to include slightly more resources than to miss potentially useful ones.

IMPORTANT GUIDELINES:
1. Be generous but not excessive - aim to include all potentially relevant resources
2. ALWAYS prioritize database tools for general queries - include as many database tools as possible
4. For wet lab sequence type of queries, ALWAYS include molecular biology tools
5. For data lake items, include datasets that could provide useful information
6. For libraries, include those that provide functions needed for analysis
7. Don't exclude resources just because they're not explicitly mentioned in the query
8. When in doubt about a database tool or molecular biology tool, include it rather than exclude it
9. If some library is useful for completing the only some part of the task, include it.
"""

        embeddings = BedrockEmbeddings(
            normalize=True, region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        rag_db_path = os.path.dirname(os.path.abspath(__file__))
        rag_db_path = os.path.join(rag_db_path, "../rag_db/system_prompt/")
        databases = {
            "tools": FAISS.load_local(
                os.path.join(rag_db_path, "tool_index"),
                embeddings,
                allow_dangerous_deserialization=True,
            ),
            "data_lake": FAISS.load_local(
                os.path.join(rag_db_path, "data_lake_index"),
                embeddings,
                allow_dangerous_deserialization=True,
            ),
            "libraries": FAISS.load_local(
                os.path.join(rag_db_path, "library_index"),
                embeddings,
                allow_dangerous_deserialization=True,
            ),
        }

        thresholds = {"tools": 0.25, "data_lake": 0.1, "libraries": 0.00}
        selected_resources = {}
        for db_name, db in databases.items():
            threshold = thresholds[db_name]
            total_docs = db.index.ntotal
            results = db.similarity_search_with_relevance_scores(
                rag_query,
                score_threshold=threshold,
                k=30,
            )
            selected_resources[db_name] = [res[0].metadata for res in results]
            print(f"{db_name}: {db.index.ntotal} -> {len(results)}")

        return selected_resources
