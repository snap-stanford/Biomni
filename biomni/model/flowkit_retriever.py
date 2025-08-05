import contextlib
import re

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


class FlowkitRetriever:
    """Retrieve tools from the flowkit functions and classes."""

    def __init__(self):
        pass

    def prompt_based_retrieval(self, query: str, resources: dict, llm=None) -> dict:
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
        prompt = f"""
You are an expert of flowkit library. Your task is to select the relevant flowkit functions and classes to help answer a user's query.

USER QUERY: {query}

Below are the available flowkit functions and classes. For each category, select items that are directly or indirectly relevant to answering the query.
Be generous in your selection - include resources that might be useful for the task, even if they're not explicitly mentioned in the query.
It's better to include slightly more resources than to miss potentially useful ones.

AVAILABLE RESOURCES:
{self._format_resources_for_prompt(resources)}

Respond with ONLY the indices of the relevant items in the following format:
[list of indices]

For example:
[0, 3, 5, 7, 9]

If there is no relevant items, use an empty list, e.g.[]

IMPORTANT GUIDELINES:
1. Be generous but not excessive - aim to include all potentially relevant resources
2. Don't exclude resources just because they're not explicitly mentioned in the query
3. When in doubt about a function or class, include it rather than exclude it
4. The maximum number of resources to select is 30.
"""

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
        selected_indices = self._parse_llm_response(response_content)
        print(f"{len(selected_indices)} resources selected")
        print(selected_indices)
        # Get the selected resources
        selected_resources = self._select_resources_from_indices(
            resources, selected_indices
        )
        return selected_resources

    def _format_resources_for_prompt(self, resources: list) -> str:
        """Format resources for inclusion in the prompt."""
        formatted = []
        index = 0

        for _, resource in enumerate(resources):
            if "methods" in resource:
                class_name = resource["name"]
                for function in resource["methods"]:
                    function_name = function["name"]
                    function_description = function["description"]
                    formatted.append(
                        f"{index}. {class_name} - {function_name}: {function_description}"
                    )
                    index += 1
            else:
                function_name = resource["name"]
                function_description = resource["description"]
                formatted.append(f"{index}. {function_name}: {function_description}")
                index += 1

        return "\n".join(formatted) if formatted else "None available"

    def _select_resources_from_indices(
        self, resources: list, selected_indices: list
    ) -> str:
        """Select resources from the list of indices."""
        index = 0
        retval = resources[:]
        for i in range(len(resources)):
            resource = resources[i]
            if "methods" in resource:
                for j in range(len(resource["methods"])):
                    if index not in selected_indices:
                        retval[i]["methods"][j] = None
                    index += 1
            else:
                if index not in selected_indices:
                    retval[i] = None
                index += 1

        retval = [r for r in retval if r is not None]
        for i in range(len(retval)):
            if "methods" in retval[i]:
                retval[i]["methods"] = [
                    retval[i]["methods"][j]
                    for j in range(len(retval[i]["methods"]))
                    if retval[i]["methods"][j] is not None
                ]

        return retval

    def _parse_llm_response(self, response: str) -> dict:
        """Parse the LLM response to extract the selected indices."""

        # Extract indices for each category
        match = re.search(r"\s*\[(.*?)\]", response, re.IGNORECASE)
        if match and match.group(1).strip():
            with contextlib.suppress(ValueError):
                return [
                    int(idx.strip()) for idx in match.group(1).split(",") if idx.strip()
                ]

        return []
