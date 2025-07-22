import json
import os

import requests


def _query_gemini_for_api(prompt, schema, system_template, api_key=None, model="gemini-2.5-flash"):
    """Helper function to query Gemini for generating API calls based on natural language prompts.

    Parameters
    ----------
    prompt (str): Natural language query to process
    schema (dict): API schema to include in the system prompt
    system_template (str): Template string for the system prompt (should have {schema} placeholder)
    api_key (str, optional): Google API key. If None, will use GOOGLE_API_KEY env variable
    model (str): Gemini model to use

    Returns
    -------
    dict: Dictionary with 'success', 'data' (if successful), 'error' (if failed), and optional 'raw_response'
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        return {"success": False, "error": f"langchain_google_genai not installed: {e}"}

    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if api_key is None:
        return {
            "success": False,
            "error": "No API key provided. Set GOOGLE_API_KEY environment variable or provide api_key parameter.",
        }

    if schema is not None:
        schema_json = json.dumps(schema, indent=2)
        system_prompt = system_template.format(schema=schema_json)
    else:
        system_prompt = system_template

    # Compose messages for Gemini
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]

    try:
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0,
            google_api_key=api_key,
        )
        response = llm.invoke(messages)
        gemini_text = response.content.strip()
        json_start = gemini_text.find("{")
        json_end = gemini_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_text = gemini_text[json_start:json_end]
            result = json.loads(json_text)
        else:
            result = json.loads(gemini_text)
        return {"success": True, "data": result, "raw_response": gemini_text}
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return {
            "success": False,
            "error": f"Failed to parse Gemini's response: {str(e)}",
            "raw_response": gemini_text if "gemini_text" in locals() else "No content found",
        }
    except Exception as e:
        return {"success": False, "error": f"Error querying Gemini: {str(e)}"}
