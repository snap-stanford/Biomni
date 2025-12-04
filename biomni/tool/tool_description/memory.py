description = [
    {
        "description": "Retrieves relevant past conversation history based on the query. "
        "Use this tool when the user asks about previous interactions, their name, personal details provided earlier, "
        "or context from past conversations.",
        "name": "retrieve_past_conversations",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "The query string to search for in the conversation history. "
                "Should be specific keywords or the user's actual question, not generic phrases.",
                "name": "query",
                "type": "str",
            }
        ],
    }
]
