import json
import os
import pickle
import time
from typing import Any

import requests
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Seq import Seq
from langchain_core.messages import HumanMessage, SystemMessage

from biomni.llm import get_llm
from biomni.utils import parse_hpo_obo


# Function to map HPO terms to names
def get_hpo_names(hpo_terms: list[str], data_lake_path: str) -> list[str]:
    """Retrieve the names of given HPO terms.

    Args:
        hpo_terms (List[str]): A list of HPO terms (e.g., ['HP:0001250']).

    Returns:
        List[str]: A list of corresponding HPO term names.

    """
    hp_dict = parse_hpo_obo(data_lake_path + "/hp.obo")

    hpo_names = []
    for term in hpo_terms:
        name = hp_dict.get(term, f"Unknown term: {term}")
        hpo_names.append(name)
    return hpo_names


def _query_llm_for_api(prompt, schema, system_template):
    """Helper function to query LLMs for generating API calls based on natural language prompts.

    Supports multiple model providers including Claude, Gemini, GPT, and others via the unified get_llm interface.

    Parameters
    ----------
    prompt (str): Natural language query to process
    schema (dict): API schema to include in the system prompt
    system_template (str): Template string for the system prompt (should have {schema} placeholder)

    Returns
    -------
    dict: Dictionary with 'success', 'data' (if successful), 'error' (if failed), and optional 'raw_response'

    """
    # Use global config for model and api_key
    try:
        from biomni.config import default_config

        model = default_config.llm
        api_key = default_config.api_key
    except ImportError:
        model = "claude-3-5-haiku-20241022"
        api_key = None

    try:
        # Format the system prompt with schema if provided
        if schema is not None:
            schema_json = json.dumps(schema, indent=2)
            system_prompt = system_template.format(schema=schema_json)
        else:
            system_prompt = system_template

        # Get LLM instance using the unified interface with config
        try:
            from biomni.config import default_config

            llm = get_llm(model=model, temperature=0.0, api_key=api_key, config=default_config)
        except ImportError:
            llm = get_llm(model=model, temperature=0.0, api_key=api_key or "EMPTY")

        # Compose messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]

        # Query the LLM
        response = llm.invoke(messages)
        llm_text = response.content.strip()

        # Find JSON boundaries (in case LLM adds explanations)
        json_start = llm_text.find("{")
        json_end = llm_text.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_text = llm_text[json_start:json_end]
            result = json.loads(json_text)
        else:
            # If no JSON found, try the whole response
            result = json.loads(llm_text)

        return {"success": True, "data": result, "raw_response": llm_text}

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return {
            "success": False,
            "error": f"Failed to parse LLM response: {str(e)}",
            "raw_response": llm_text if "llm_text" in locals() else "No content found",
        }
    except Exception as e:
        return {"success": False, "error": f"Error querying LLM: {str(e)}"}


def _query_rest_api(endpoint, method="GET", params=None, headers=None, json_data=None, description=None):
    """General helper function to query REST APIs with consistent error handling.

    Parameters
    ----------
    endpoint (str): Full URL endpoint to query
    method (str): HTTP method ("GET" or "POST")
    params (dict, optional): Query parameters to include in the URL
    headers (dict, optional): HTTP headers for the request
    json_data (dict, optional): JSON data for POST requests
    description (str, optional): Description of this query for error messages

    Returns
    -------
    dict: Dictionary containing the result or error information

    """
    # Set default headers if not provided
    if headers is None:
        headers = {"Accept": "application/json"}

    # Set default description if not provided
    if description is None:
        description = f"{method} request to {endpoint}"

    url_error = None

    try:
        # Make the API request
        if method.upper() == "GET":
            response = requests.get(endpoint, params=params, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(endpoint, params=params, headers=headers, json=json_data)
        else:
            return {"error": f"Unsupported HTTP method: {method}"}

        url_error = str(response.text)
        response.raise_for_status()

        # Try to parse JSON response
        try:
            result = response.json()
        except ValueError:
            # Return raw text if not JSON
            result = {"raw_text": response.text}

        return {
            "success": True,
            "query_info": {
                "endpoint": endpoint,
                "method": method,
                "description": description,
            },
            "result": result,
        }

    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        response_text = ""

        # Try to get more detailed error info from response
        if hasattr(e, "response") and e.response:
            try:
                error_json = e.response.json()
                if "messages" in error_json:
                    error_msg = "; ".join(error_json["messages"])
                elif "message" in error_json:
                    error_msg = error_json["message"]
                elif "error" in error_json:
                    error_msg = error_json["error"]
                elif "detail" in error_json:
                    error_msg = error_json["detail"]
            except Exception:
                response_text = e.response.text

        return {
            "success": False,
            "error": f"API error: {error_msg}",
            "query_info": {
                "endpoint": endpoint,
                "method": method,
                "description": description,
            },
            "response_url_error": url_error,
            "response_text": response_text,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}",
            "query_info": {
                "endpoint": endpoint,
                "method": method,
                "description": description,
            },
        }


def _query_ncbi_database(
    database: str,
    search_term: str,
    result_formatter=None,
    max_results: int = 3,
) -> dict[str, Any]:
    """Core function to query NCBI databases using Claude for query interpretation and NCBI eutils.

    Parameters
    ----------
    database (str): NCBI database to query (e.g., "clinvar", "gds", "geoprofiles")
    result_formatter (callable): Function to format results from the database
    api_key (str): Anthropic API key. If None, will look for ANTHROPIC_API_KEY environment variable
    model (str): Anthropic model to use
    max_results (int): Maximum number of results to return
    verbose (bool): Whether to return verbose results

    Returns
    -------
    dict: Dictionary containing both the structured query and the results

    """
    # Query NCBI API using the structured search term
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esearch_params = {
        "db": database,
        "term": search_term,
        "retmode": "json",
        "retmax": 100,
        "usehistory": "y",  # Use history server to store results
    }

    # Get IDs of matching entries
    search_response = _query_rest_api(
        endpoint=esearch_url,
        method="GET",
        params=esearch_params,
        description="NCBI ESearch API query",
    )

    if not search_response["success"]:
        return search_response

    search_data = search_response["result"]

    # If we have results, fetch the details
    if "esearchresult" in search_data and int(search_data["esearchresult"]["count"]) > 0:
        # Extract WebEnv and query_key from the search results
        webenv = search_data["esearchresult"].get("webenv", "")
        query_key = search_data["esearchresult"].get("querykey", "")

        # Use WebEnv and query_key if available
        if webenv and query_key:
            # Get details using eSummary
            esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            esummary_params = {
                "db": database,
                "query_key": query_key,
                "WebEnv": webenv,
                "retmode": "json",
                "retmax": max_results,
            }

            details_response = _query_rest_api(
                endpoint=esummary_url,
                method="GET",
                params=esummary_params,
                description="NCBI ESummary API query",
            )

            if not details_response["success"]:
                return details_response

            results = details_response["result"]

        else:
            # Fall back to direct ID fetch
            id_list = search_data["esearchresult"]["idlist"][:max_results]

            # Get details for each ID
            esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            esummary_params = {
                "db": database,
                "id": ",".join(id_list),
                "retmode": "json",
            }

            details_response = _query_rest_api(
                endpoint=esummary_url,
                method="GET",
                params=esummary_params,
                description="NCBI ESummary API query",
            )

            if not details_response["success"]:
                return details_response

            results = details_response["result"]

        # Format results using the provided formatter
        formatted_results = result_formatter(results) if result_formatter else results

        # Return the combined information
        return {
            "database": database,
            "query_interpretation": search_term,
            "total_results": int(search_data["esearchresult"]["count"]),
            "formatted_results": formatted_results,
        }
    else:
        return {
            "database": database,
            "query_interpretation": search_term,
            "total_results": 0,
            "formatted_results": [],
        }


def _format_query_results(result, options=None):
    """A general-purpose formatter for query function results to reduce output size.

    Parameters
    ----------
    result (dict): The original API response dictionary
    options (dict, optional): Formatting options including:
        - max_items (int): Maximum number of items to include in lists (default: 5)
        - max_depth (int): Maximum depth to traverse in nested dictionaries (default: 2)
        - include_keys (list): Only include these top-level keys (overrides exclude_keys)
        - exclude_keys (list): Exclude these keys from the output
        - summarize_lists (bool): Whether to summarize long lists (default: True)
        - truncate_strings (int): Maximum length for string values (default: 100)

    Returns
    -------
    dict: A condensed version of the input results

    """

    def _format_value(value, depth, options):
        """Recursively format a value based on its type and formatting options.

        Parameters
        ----------
        value: The value to format
        depth (int): Current recursion depth
        options (dict): Formatting options

        Returns
        -------
        Formatted value

        """
        # Base case: reached max depth
        if depth >= options["max_depth"] and (isinstance(value, dict | list)):
            if isinstance(value, dict):
                return {
                    "_summary": f"Nested dictionary with {len(value)} keys",
                    "_keys": list(value.keys())[: options["max_items"]],
                }
            else:  # list
                return _summarize_list(value, options)

        # Process based on type
        if isinstance(value, dict):
            return _format_dict(value, depth, options)
        elif isinstance(value, list):
            return _format_list(value, depth, options)
        elif isinstance(value, str) and len(value) > options["truncate_strings"]:
            return value[: options["truncate_strings"]] + "... (truncated)"
        else:
            return value

    def _format_dict(d, depth, options):
        """Format a dictionary according to options."""
        result = {}

        # Filter keys based on include/exclude options
        keys_to_process = d.keys()
        if depth == 0 and options["include_keys"]:  # Only apply at top level
            keys_to_process = [k for k in keys_to_process if k in options["include_keys"]]
        elif depth == 0 and options["exclude_keys"]:  # Only apply at top level
            keys_to_process = [k for k in keys_to_process if k not in options["exclude_keys"]]

        # Process each key
        for key in keys_to_process:
            result[key] = _format_value(d[key], depth + 1, options)

        return result

    def _format_list(lst, depth, options):
        """Format a list according to options."""
        if options["summarize_lists"] and len(lst) > options["max_items"]:
            return _summarize_list(lst, options)

        result = []
        for i, item in enumerate(lst):
            if i >= options["max_items"]:
                remaining = len(lst) - options["max_items"]
                result.append(f"... {remaining} more items (omitted)")
                break
            result.append(_format_value(item, depth + 1, options))

        return result

    def _summarize_list(lst, options):
        """Create a summary for a list."""
        if not lst:
            return []

        # Sample a few items
        sample = lst[: min(3, len(lst))]
        sample_formatted = [_format_value(item, options["max_depth"], options) for item in sample]

        # For homogeneous lists, provide type info
        if len(lst) > 0:
            item_type = type(lst[0]).__name__
            homogeneous = all(isinstance(item, type(lst[0])) for item in lst)
            type_info = f"all {item_type}" if homogeneous else "mixed types"
        else:
            type_info = "empty"

        return {
            "_summary": f"List with {len(lst)} items ({type_info})",
            "_sample": sample_formatted,
        }

    if options is None:
        options = {}

    # Default options
    default_options = {
        "max_items": 5,
        "max_depth": 20,
        "include_keys": None,
        "exclude_keys": ["raw_response", "debug_info", "request_details"],
        "summarize_lists": True,
        "truncate_strings": 100,
    }

    # Merge provided options with defaults
    for key, value in default_options.items():
        if key not in options:
            options[key] = value

    # Filter and format the result
    formatted = _format_value(result, 0, options)
    return formatted


def query_uniprot(
    prompt=None,
    endpoint=None,
    max_results=5,
):
    """Query the UniProt REST API using either natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about proteins (e.g., "Find information about human insulin")
    endpoint (str, optional): Full or partial UniProt API endpoint URL to query directly
                            (e.g., "https://rest.uniprot.org/uniprotkb/P01308")
    max_results (int): Maximum number of results to return

    Returns
    -------
    dict: Dictionary containing the query information and the UniProt API results

    Examples
    --------
    - Natural language: query_uniprot(prompt="Find information about human insulin protein")
    - Direct endpoint: query_uniprot(endpoint="https://rest.uniprot.org/uniprotkb/P01308")

    """
    # Base URL for UniProt API
    base_url = "https://rest.uniprot.org"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load UniProt schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "uniprot.pkl")
        with open(schema_path, "rb") as f:
            uniprot_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a protein biology expert specialized in using the UniProt REST API.

        Based on the user's natural language request, determine the appropriate UniProt REST API endpoint and parameters.

        UNIPROT REST API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including base URL, dataset, endpoint type, and parameters)
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - Base URL is "https://rest.uniprot.org"
        - Search in reviewed (Swiss-Prot) entries first before using non-reviewed (TrEMBL) entries
        - Assume organism is human unless otherwise specified. Human taxonomy ID is 9606
        - Use gene_exact: for exact gene name searches
        - Use specific query fields like accession:, gene:, organism_id: in search queries
        - Use quotes for terms with spaces: organism_name:"Homo sapiens"

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=uniprot_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Use provided endpoint directly
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        description = "Direct query to provided endpoint"

    # Use the common REST API helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    return api_result


def query_alphafold(
    uniprot_id,
    endpoint="prediction",
    residue_range=None,
    download=False,
    output_dir=None,
    file_format="pdb",
    model_version="v4",
    model_number=1,
):
    """Query the AlphaFold Database API for protein structure predictions.

    Parameters
    ----------
    uniprot_id (str): UniProt accession ID (e.g., "P12345")
    endpoint (str, optional): Specific AlphaFold API endpoint to query:
                            "prediction", "summary", or "annotations"
    residue_range (str, optional): Specific residue range in format "start-end" (e.g., "1-100")
    download (bool): Whether to download structure files
    output_dir (str, optional): Directory to save downloaded files (default: current directory)
    file_format (str): Format of the structure file to download - "pdb" or "cif"
    model_version (str): AlphaFold model version - "v4" (latest) or "v3", "v2", "v1"
    model_number (int): Model number (1-5, with 1 being the highest confidence model)

    Returns
    -------
    dict: Dictionary containing both the query information and the AlphaFold results

    Examples
    --------
    - Basic query: query_alphafold(uniprot_id="P53_HUMAN")
    - Download structure: query_alphafold(uniprot_id="P53_HUMAN", download=True, output_dir="./structures")
    - Get annotations: query_alphafold(uniprot_id="P53_HUMAN", endpoint="annotations")

    """
    # Base URL for AlphaFold API
    base_url = "https://alphafold.ebi.ac.uk/api"

    # Ensure we have a UniProt ID
    if not uniprot_id:
        return {"error": "UniProt ID is required"}

    # Validate endpoint
    valid_endpoints = ["prediction", "summary", "annotations"]
    if endpoint not in valid_endpoints:
        return {"error": f"Invalid endpoint. Must be one of: {', '.join(valid_endpoints)}"}

    # Construct the API URL based on endpoint
    if endpoint == "prediction":
        url = f"{base_url}/prediction/{uniprot_id}"
    elif endpoint == "summary":
        url = f"{base_url}/uniprot/summary/{uniprot_id}.json"
    elif endpoint == "annotations":
        if residue_range:
            url = f"{base_url}/annotations/{uniprot_id}/{residue_range}"
        else:
            url = f"{base_url}/annotations/{uniprot_id}"

    try:
        # Make the API request
        response = requests.get(url)
        response.raise_for_status()

        # Parse the response as JSON
        result = response.json()

        # Handle download request if specified
        download_info = None
        if download:
            # Ensure output directory exists
            if not output_dir:
                output_dir = "."
            os.makedirs(output_dir, exist_ok=True)

            # Generate standard AlphaFold filename
            file_ext = file_format.lower()
            filename = f"AF-{uniprot_id}-F{model_number}-model_{model_version}.{file_ext}"
            file_path = os.path.join(output_dir, filename)

            # Construct download URL
            download_url = f"https://alphafold.ebi.ac.uk/files/{filename}"

            # Download the file
            download_response = requests.get(download_url)
            if download_response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(download_response.content)
                download_info = {
                    "success": True,
                    "file_path": file_path,
                    "url": download_url,
                }
            else:
                download_info = {
                    "success": False,
                    "error": f"Failed to download file (status code: {download_response.status_code})",
                    "url": download_url,
                }

        # Return the query information and results
        response_data = {
            "query_info": {
                "uniprot_id": uniprot_id,
                "endpoint": endpoint,
                "residue_range": residue_range,
                "url": url,
            },
            "result": result,
        }

        if download_info:
            response_data["download"] = download_info

        return response_data

    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        response_text = ""

        # Try to get more detailed error info from response
        if hasattr(e, "response") and e.response:
            try:
                error_json = e.response.json()
                if "message" in error_json:
                    error_msg = error_json["message"]
            except Exception:
                response_text = e.response.text

        return {
            "error": f"AlphaFold API error: {error_msg}",
            "query_info": {
                "uniprot_id": uniprot_id,
                "endpoint": endpoint,
                "residue_range": residue_range,
                "url": url,
            },
            "response_text": response_text,
        }
    except Exception as e:
        return {
            "error": f"Error: {str(e)}",
            "query_info": {
                "uniprot_id": uniprot_id,
                "endpoint": endpoint,
                "residue_range": residue_range,
            },
        }


def query_interpro(
    prompt=None,
    endpoint=None,
    max_results=3,
):
    """Query the InterPro REST API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about protein domains or families
    endpoint (str, optional): Direct endpoint path or full URL (e.g., "/entry/interpro/IPR023411"
                             or "https://www.ebi.ac.uk/interpro/api/entry/interpro/IPR023411")
    max_results (int): Maximum number of results to return per page

    Returns
    -------
    dict: Dictionary containing both the query information and the InterPro API results

    Examples
    --------
    - Natural language: query_interpro("Find information about kinase domains in InterPro")
    - Direct endpoint: query_interpro(endpoint="/entry/interpro/IPR023411")

    """
    # Base URL for InterPro API
    base_url = "https://www.ebi.ac.uk/interpro/api"

    # Default parameters
    format = "json"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load InterPro schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "interpro.pkl")
        with open(schema_path, "rb") as f:
            interpro_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a protein domain expert specialized in using the InterPro REST API.

        Based on the user's natural language request, determine the appropriate InterPro REST API endpoint.

        INTERPRO REST API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://www.ebi.ac.uk/interpro/api")
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - Path components for data types: entry, protein, structure, set, taxonomy, proteome
        - Common sources: interpro, pfam, cdd, uniprot, pdb
        - Protein subtypes can be "reviewed" or "unreviewed"
        - For specific entries, use lowercase accessions (e.g., "ipr000001" instead of "IPR000001")
        - Endpoints can be hierarchical like "/entry/interpro/protein/uniprot/P04637"

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=interpro_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Extract the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        # If it's just a path, add the base URL
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"

        description = "Direct query to provided endpoint"

    # Add pagination parameters
    params = {"page": 1, "page_size": max_results}

    # Add format parameter if not json
    if format and format != "json":
        params["format"] = format

    # Make the API request
    api_result = _query_rest_api(endpoint=endpoint, method="GET", params=params, description=description)

    return api_result


def query_pdb(
    prompt=None,
    query=None,
    max_results=3,
):
    """Query the RCSB PDB database using natural language or a direct structured query.

    Parameters
    ----------
    prompt (str, required): Natural language query about protein structures
    query (dict, optional): Direct structured query in RCSB Search API format (overrides prompt)
    max_results (int): Maximum number of results to return

    Returns
    -------
    dict: Dictionary containing the structured query, search results, and identifiers

    Examples
    --------
    - Natural language: query_pdb("Find structures of human insulin")
    - Direct query: query_pdb(query={"query": {"type": "terminal", "service": "full_text",
                           "parameters": {"value": "insulin"}}, "return_type": "entry"})

    """
    # Default parameters
    return_type = "entry"
    search_service = "full_text"

    # Generate search query from natural language if prompt is provided and query is not
    if prompt and not query:
        # Load schema from pickle file
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "pdb.pkl")

        with open(schema_path, "rb") as f:
            schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a structural biology expert that creates precise RCSB PDB Search API queries based on natural language requests.

        SEARCH API SCHEMA:
        {schema}

        IMPORTANT GUIDELINES:
        1. Choose the appropriate search_service based on the query:
           - Use "text" for attribute-specific searches (REQUIRES attribute, operator, and value)
           - Use "full_text" for general keyword searches across multiple fields
           - Use appropriate specialized services for sequence, structure, motif searches

        2. For "text" searches, you MUST specify:
           - attribute: The specific field to search (use common_attributes from schema)
           - operator: The comparison method (exact_match, contains_words, less_or_equal, etc.)
           - value: The search term or value

        3. For "full_text" searches, only specify:
           - value: The search term(s)

        4. For combined searches, use "group" nodes with logical_operator ("and" or "or")

        5. Always specify the appropriate return_type based on what the user is looking for

        Generate a well-formed Search API query JSON object. Return ONLY the JSON with no additional explanation.
        """

        # Query Claude to generate the search query
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return {
                "error": llm_result["error"],
                "llm_response": llm_result.get("raw_response", "No response"),
            }

        # Get the query from Claude's response
        query_json = llm_result["data"]
    else:
        # Use provided query directly
        query_json = (
            query
            if query
            else {
                "query": {
                    "type": "terminal",
                    "service": search_service,
                    "parameters": {"value": prompt},
                },
                "return_type": return_type,
            }
        )

    # Ensure return_type is set
    if "return_type" not in query_json:
        query_json["return_type"] = return_type

    # Add request options for pagination
    if "request_options" not in query_json:
        query_json["request_options"] = {}

    if "paginate" not in query_json["request_options"]:
        query_json["request_options"]["paginate"] = {"start": 0, "rows": max_results}

    # Use query_rest_api to execute the search
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    api_result = _query_rest_api(
        endpoint=search_url,
        method="POST",
        json_data=query_json,
        description="PDB Search API query",
    )

    return api_result


def query_pdb_identifiers(identifiers, return_type="entry", download=False, attributes=None):
    """Retrieve detailed data and/or download files for PDB identifiers.

    Parameters
    ----------
    identifiers (list): List of PDB identifiers (from query_pdb)
    return_type (str): Type of results: "entry", "assembly", "polymer_entity", etc.
    download (bool): Whether to download PDB structure files
    attributes (list, optional): List of specific attributes to retrieve

    Returns
    -------
    dict: Dictionary containing the detailed data and file paths if downloaded

    Example:
    - Search and then get details:
        results = query_pdb("Find structures of human insulin")
        details = get_pdb_details(results["identifiers"], download=True)

    """
    if not identifiers:
        return {"error": "No identifiers provided"}

    try:
        # Fetch detailed data using Data API
        detailed_results = []
        for identifier in identifiers:
            try:
                # Determine the appropriate endpoint based on return_type and identifier format
                if return_type == "entry":
                    data_url = f"https://data.rcsb.org/rest/v1/core/entry/{identifier}"
                elif return_type == "polymer_entity":
                    entry_id, entity_id = identifier.split("_")
                    data_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{entry_id}/{entity_id}"
                elif return_type == "nonpolymer_entity":
                    entry_id, entity_id = identifier.split("_")
                    data_url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{entry_id}/{entity_id}"
                elif return_type == "polymer_instance":
                    entry_id, asym_id = identifier.split(".")
                    data_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity_instance/{entry_id}/{asym_id}"
                elif return_type == "assembly":
                    entry_id, assembly_id = identifier.split("-")
                    data_url = f"https://data.rcsb.org/rest/v1/core/assembly/{entry_id}/{assembly_id}"
                elif return_type == "mol_definition":
                    data_url = f"https://data.rcsb.org/rest/v1/core/chem_comp/{identifier}"

                # Fetch data
                data_response = requests.get(data_url)
                data_response.raise_for_status()
                entity_data = data_response.json()

                # Filter attributes if specified
                if attributes:
                    filtered_data = {}
                    for attr in attributes:
                        parts = attr.split(".")
                        current = entity_data
                        try:
                            for part in parts[:-1]:
                                current = current[part]
                            filtered_data[attr] = current[parts[-1]]
                        except (KeyError, TypeError):
                            filtered_data[attr] = None
                    entity_data = filtered_data

                detailed_results.append({"identifier": identifier, "data": entity_data})
            except Exception as e:
                detailed_results.append({"identifier": identifier, "error": str(e)})

        # Download structure files if requested
        if download:
            for identifier in identifiers:
                if "_" in identifier or "." in identifier or "-" in identifier:
                    # For non-entry identifiers, extract the PDB ID
                    if "_" in identifier:
                        pdb_id = identifier.split("_")[0]
                    elif "." in identifier:
                        pdb_id = identifier.split(".")[0]
                    elif "-" in identifier:
                        pdb_id = identifier.split("-")[0]
                else:
                    pdb_id = identifier

                try:
                    # Download PDB file
                    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                    pdb_response = requests.get(pdb_url)

                    if pdb_response.status_code == 200:
                        # Create data directory if it doesn't exist
                        data_dir = os.path.join(os.path.dirname(__file__), "data", "pdb")
                        os.makedirs(data_dir, exist_ok=True)

                        # Save PDB file
                        pdb_file_path = os.path.join(data_dir, f"{pdb_id}.pdb")
                        with open(pdb_file_path, "wb") as pdb_file:
                            pdb_file.write(pdb_response.content)

                        # Add download information to results
                        for result in detailed_results:
                            if result["identifier"] == identifier or result["identifier"].startswith(pdb_id):
                                result["pdb_file_path"] = pdb_file_path
                except Exception as e:
                    for result in detailed_results:
                        if result["identifier"] == identifier or result["identifier"].startswith(pdb_id):
                            result["download_error"] = str(e)

        return {"detailed_results": detailed_results}

    except Exception as e:
        return {"error": f"Error retrieving PDB details: {str(e)}"}


def query_kegg(prompt, endpoint=None, verbose=True):
    """Take a natural language prompt and convert it to a structured KEGG API query.

    Parameters
    ----------
    prompt (str): Natural language query about KEGG data (e.g., "Find human pathways related to glycolysis")
    endpoint (str, optional): Direct KEGG API endpoint to query
    verbose (bool): Whether to print verbose output

    Returns
    -------
    dict: Dictionary containing both the structured query and the KEGG results

    """
    base_url = "https://rest.kegg.jp"

    if not prompt and not endpoint:
        return {"error": "Either a prompt or an endpoint must be provided"}

    if prompt:
        # Load schema from pickle file
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "kegg.pkl")
        with open(schema_path, "rb") as f:
            kegg_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a bioinformatics expert that helps convert natural language queries into KEGG API requests.

        Based on the user's natural language request, you will generate a structured query for the KEGG API.

        The KEGG API has the following general form:
        https://rest.kegg.jp/<operation>/<argument>[/<argument2>[/<argument3> ...]]

        Where <operation> can be one of: info, list, find, get, conv, link, ddi

        Here is the schema of available operations, databases, and other details:
        {schema}

        Output only a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://rest.kegg.jp")
        2. "description": A brief description of what the query is doing

        IMPORTANT: Your response must ONLY contain a JSON object with the required fields.

        EXAMPLES OF CORRECT OUTPUTS:
        - For "Find information about glycolysis pathway": {{"full_url": "https://rest.kegg.jp/info/pathway/hsa00010", "description": "Finding information about the glycolysis pathway"}}
        - For "Get information about the human BRCA1 gene": {{"full_url": "https://rest.kegg.jp/get/hsa:672", "description": "Retrieving information about BRCA1 gene in human"}}
        - For "List all human pathways": {{"full_url": "https://rest.kegg.jp/list/pathway/hsa", "description": "Listing all human-specific pathways"}}
        - For "Convert NCBI gene ID 672 to KEGG ID": {{"full_url": "https://rest.kegg.jp/conv/genes/ncbi-geneid:672", "description": "Converting NCBI Gene ID 672 to KEGG gene identifier"}}
        """

        # Query LLM to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=kegg_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

            # Extract the query info from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info["full_url"]
        description = query_info["description"]

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }

    if endpoint:
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        description = "Direct query to KEGG API"

    # Execute the KEGG API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def query_stringdb(
    prompt=None,
    endpoint=None,
    download_image=False,
    output_dir=None,
    verbose=True,
):
    """Query the STRING protein interaction database using natural language or direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about protein interactions
    endpoint (str, optional): Full URL to query directly (overrides prompt)
    download_image (bool): Whether to download image results (for image endpoints)
    output_dir (str, optional): Directory to save downloaded files (default: current directory)

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_stringdb("Show protein interactions for BRCA1 and BRCA2 in humans")
    - Direct endpoint: query_stringdb(endpoint="https://string-db.org/api/json/network?identifiers=BRCA1,BRCA2&species=9606")

    """
    # Base URL for STRING API
    base_url = "https://version-12-0.string-db.org/api"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load STRING schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "stringdb.pkl")
        with open(schema_path, "rb") as f:
            stringdb_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a protein interaction expert specialized in using the STRING database API.

        Based on the user's natural language request, determine the appropriate STRING API endpoint and parameters.

        STRING API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including all parameters)
        2. "description": A brief description of what the query is doing
        3. "output_format": The format of the output (json, tsv, image, svg)

        SPECIAL NOTES:
        - Common species IDs: 9606 (human), 10090 (mouse), 7227 (fruit fly), 4932 (yeast)
        - For protein identifiers, use either gene names (e.g., "BRCA1") or UniProt IDs (e.g., "P38398")
        - The "required_score" parameter accepts values from 0 to 1000 (higher means more stringent)
        - Add "caller_identity=bioagentos_api" as a parameter

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=stringdb_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")
        output_format = query_info.get("output_format", "json")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Use direct endpoint
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        description = "Direct query to STRING API"
        output_format = "json"

        # Try to determine output format from URL
        if "image" in endpoint or "svg" in endpoint:
            output_format = "image"

    # Check if we're dealing with an image request
    is_image = output_format in ["image", "highres_image", "svg"]

    if is_image:
        if download_image:
            # For images, we need to handle the download manually
            try:
                response = requests.get(endpoint, stream=True)
                response.raise_for_status()

                # Create output directory if needed
                if not output_dir:
                    output_dir = "."
                os.makedirs(output_dir, exist_ok=True)

                # Generate filename based on endpoint
                endpoint_parts = endpoint.split("/")
                filename = f"string_{endpoint_parts[-2]}_{int(time.time())}.{output_format}"
                file_path = os.path.join(output_dir, filename)

                # Save the image
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)

                return {
                    "success": True,
                    "query_info": {
                        "endpoint": endpoint,
                        "description": description,
                        "output_format": output_format,
                    },
                    "result": {
                        "image_saved": True,
                        "file_path": file_path,
                        "content_type": response.headers.get("Content-Type"),
                    },
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error downloading image: {str(e)}",
                    "query_info": {"endpoint": endpoint, "description": description},
                }
        else:
            # Just report that an image is available but not downloaded
            return {
                "success": True,
                "query_info": {
                    "endpoint": endpoint,
                    "description": description,
                    "output_format": output_format,
                },
                "result": {
                    "image_available": True,
                    "download_url": endpoint,
                    "note": "Set download_image=True to save the image",
                },
            }

    # For non-image requests, use the REST API helper
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def query_iucn(
    prompt=None,
    endpoint=None,
    token="",
    verbose=True,
):
    """Query the IUCN Red List API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about species conservation status
    endpoint (str, optional): API endpoint name (e.g., "species/id/12392") or full URL
    token (str): IUCN API token - required for all queries
    verbose (bool): Whether to print verbose output

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_iucn("Get conservation status of white rhinoceros", token="your-token")
    - Direct endpoint: query_iucn(endpoint="species/id/12392", token="your-token")

    """
    # Base URL for IUCN API
    base_url = "https://apiv3.iucnredlist.org/api/v3"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # Ensure we have a token
    if not token:
        return {"error": "IUCN API token is required. Get one at https://apiv3.iucnredlist.org/api/v3/token"}

    # If using prompt, parse with Claude
    if prompt:
        # Load IUCN schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "iucn.pkl")
        with open(schema_path, "rb") as f:
            iucn_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a conservation biology expert specialized in using the IUCN Red List API.

        Based on the user's natural language request, determine the appropriate IUCN API endpoint.

        IUCN API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://apiv3.iucnredlist.org/api/v3" and any path parameters)
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - The token parameter will be added automatically, do not include it in your URL
        - For taxonomic queries, prefer using scientific names over common names
        - For region-specific queries, use region identifiers from the schema
        - For species queries, try to use the species ID if known, otherwise use scientific name

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=iucn_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        if not endpoint.startswith("http"):
            endpoint = f"{base_url}{endpoint}" if endpoint.startswith("/") else f"{base_url}/{endpoint}"
        description = "Direct query to IUCN API"

    # Add token as query parameter
    params = {"token": token}

    # Execute the IUCN API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", params=params, description=description)

    # For security, remove token from the results
    if "query_info" in api_result and "endpoint" in api_result["query_info"]:
        api_result["query_info"]["endpoint"] = api_result["query_info"]["endpoint"].replace(token, "TOKEN_HIDDEN")

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def query_paleobiology(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the Paleobiology Database (PBDB) API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about fossil records
    endpoint (str, optional): API endpoint name or full URL
    verbose (bool): Whether to print verbose output

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_paleobiology("Find fossil records of Tyrannosaurus rex")
    - Direct endpoint: query_paleobiology(endpoint="data1.2/taxa/list.json?name=Tyrannosaurus")

    """
    # Base URL for PBDB API
    base_url = "https://paleobiodb.org/data1.2"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load PBDB schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "pbdb.pkl")
        with open(schema_path, "rb") as f:
            pbdb_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a paleobiology expert specialized in using the Paleobiology Database (PBDB) API.

        Based on the user's natural language request, determine the appropriate PBDB API endpoint and parameters.

        PBDB API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://paleobiodb.org/data1.2" and format extension)
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - For taxonomic queries, be specific about taxonomic ranks and names
        - For geographic queries, use standard country/continent names or coordinate bounding boxes
        - For time interval queries, use standard geological time names (e.g., "Cretaceous", "Maastrichtian")
        - Use appropriate format extension (.json, .txt, .csv, .tsv) based on the query
        - If appropriate, use "vocab=pbdb" (default) or "vocab=com" (compact) parameter in the URL
        - For detailed occurrence data, include "show=paleoloc,phylo" in the parameters

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=pbdb_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        if not endpoint.startswith("http"):
            # Add base URL if it's just a path
            endpoint = f"{base_url}/{endpoint}" if not endpoint.startswith("/") else f"{base_url}{endpoint}"

        description = "Direct query to PBDB API"

    # Check if we're dealing with an image request
    is_image = endpoint.endswith(".png")

    if is_image:
        # For image queries, we need special handling
        try:
            response = requests.get(endpoint)
            response.raise_for_status()

            # Return image metadata without the binary data
            return {
                "success": True,
                "query_info": {
                    "endpoint": endpoint,
                    "description": description,
                    "format": "png",
                },
                "result": {
                    "content_type": response.headers.get("Content-Type"),
                    "size_bytes": len(response.content),
                    "note": "Binary image data not included in response",
                },
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error retrieving image: {str(e)}",
                "query_info": {"endpoint": endpoint, "description": description},
            }

    # For non-image requests, use the REST API helper
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def query_jaspar(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the JASPAR REST API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about transcription factor binding profiles
    endpoint (str, optional): API endpoint path (e.g., "/matrix/MA0002.2/") or full URL
    verbose (bool): Whether to print verbose output

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_jaspar("Find all transcription factor matrices for human")
    - Direct endpoint: query_jaspar(endpoint="/matrix/MA0002.2/")

    """
    # Base URL for JASPAR API
    base_url = "https://jaspar.elixir.no/api/v1"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load JASPAR schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "jaspar.pkl")
        with open(schema_path, "rb") as f:
            jaspar_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a transcription factor binding site expert specialized in using the JASPAR REST API.

        Based on the user's natural language request, determine the appropriate JASPAR REST API endpoint and parameters.

        JASPAR REST API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://jaspar.elixir.no/api/v1" and any parameters)
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - Common taxonomic groups include: vertebrates, plants, fungi, insects, nematodes, urochordates
        - Common collections include: CORE, UNVALIDATED, PENDING, etc.
        - Matrix IDs follow the format MA####.# (e.g., MA0002.2)
        - For inferring matrices from sequences, provide the protein sequence directly in the path

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=jaspar_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        if not endpoint.startswith("http"):
            # Clean up endpoint format
            if not endpoint.startswith("/"):
                endpoint = "/" + endpoint

            # Ensure endpoint ends with /
            if not endpoint.endswith("/"):
                endpoint = endpoint + "/"

            # Add base URL
            endpoint = f"{base_url}{endpoint}"

        description = "Direct query to JASPAR API"

    # Execute the JASPAR API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def query_worms(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the World Register of Marine Species (WoRMS) REST API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about marine species
    endpoint (str, optional): Full URL or endpoint specification
    verbose (bool): Whether to print verbose output

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_worms("Find information about the blue whale")
    - Direct endpoint: query_worms(endpoint="https://www.marinespecies.org/rest/AphiaRecordByName/Balaenoptera%20musculus")

    """
    # Base URL for WoRMS API
    base_url = "https://www.marinespecies.org/rest"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load WoRMS schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "worms.pkl")
        with open(schema_path, "rb") as f:
            worms_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a marine biology expert specialized in using the World Register of Marine Species (WoRMS) API.

        Based on the user's natural language request, determine the appropriate WoRMS API endpoint and parameters.

        WORMS API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://www.marinespecies.org/rest" and any path/query parameters)
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - For taxonomic searches, be precise with scientific names and use proper capitalization
        - For fuzzy matching, include "fuzzy=true" in the URL query parameters
        - When searching by name, prefer "AphiaRecordByName" for exact matches and "AphiaRecordsByName" for broader results
        - AphiaID is the main identifier in WoRMS (e.g., Blue Whale is 137087)
        - For multiple IDs or names, use the appropriate POST endpoint

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=worms_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL and details from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        if not endpoint.startswith("http"):
            # Add base URL if it's just a path
            endpoint = f"{base_url}/{endpoint}" if not endpoint.startswith("/") else f"{base_url}{endpoint}"

        description = "Direct query to WoRMS API"

    # Execute the WoRMS API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def query_cbioportal(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the cBioPortal REST API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about cancer genomics data
    endpoint (str, optional): API endpoint path (e.g., "/studies/brca_tcga/patients") or full URL
    verbose (bool): Whether to print verbose output

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_cbioportal("Find mutations in BRCA1 for breast cancer")
    - Direct endpoint: query_cbioportal(endpoint="/studies/brca_tcga/molecular-profiles")

    """
    # Base URL for cBioPortal API
    base_url = "https://www.cbioportal.org/api"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load cBioPortal schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "cbioportal.pkl")
        with open(schema_path, "rb") as f:
            cbioportal_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a cancer genomics expert specialized in using the cBioPortal REST API.

        Based on the user's natural language request, determine the appropriate cBioPortal REST API endpoint and parameters.

        CBIOPORTAL REST API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://www.cbioportal.org/api" and any parameters)
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - For gene queries, use either Hugo symbol (e.g., "BRCA1") or Entrez ID (e.g., 672)
        - For pagination, include parameters "pageNumber" and "pageSize" if needed
        - For mutation data queries, always include appropriate sample identifiers
        - Common studies include: "brca_tcga" (breast cancer), "gbm_tcga" (glioblastoma), "luad_tcga" (lung adenocarcinoma)
        - For molecular profiles, common IDs follow pattern: "[study]_[data_type]" (e.g., "brca_tcga_mutations")
        - Consider including "projection=DETAILED" for more comprehensive results when appropriate

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=cbioportal_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        if not endpoint.startswith("http"):
            # Clean up endpoint format
            if not endpoint.startswith("/"):
                endpoint = "/" + endpoint

            # Add base URL
            endpoint = f"{base_url}{endpoint}"

        description = "Direct query to cBioPortal API"

    # Execute the cBioPortal API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def query_clinvar(
    prompt=None,
    search_term=None,
    max_results=3,
):
    """Take a natural language prompt and convert it to a structured ClinVar query.

    Parameters
    ----------
    prompt (str): Natural language query about genetic variants (e.g., "Find pathogenic BRCA1 variants")
    search_term (str): Direct search term in ClinVar syntax
    max_results (int): Maximum number of results to return

    Returns
    -------
    dict: Dictionary containing both the structured query and the ClinVar results

    """
    if not prompt and not search_term:
        return {"error": "Either a prompt or an endpoint must be provided"}

    if prompt:
        # Load ClinVar schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "clinvar.pkl")
        with open(schema_path, "rb") as f:
            clinvar_schema = pickle.load(f)

        # ClinVar system prompt template
        system_prompt_template = """
        You are a genetics research assistant that helps convert natural language queries into structured ClinVar search queries.

        Based on the user's natural language request, you will generate a structured search for the ClinVar database.

        Output only a JSON object with the following fields:
        1. "search_term": The exact search query to use with the ClinVar API

        IMPORTANT: Your response must ONLY contain a JSON object with the search term field.

        Your "search_term" MUST strictly follow these ClinVar search syntax rules/tags:

        {schema}

        For combining terms: Use AND, OR, NOT (must be capitalized)
        For complex logic: Use parentheses
        For terms with multiple words: use double quotes escaped with a backslash or underscore (e.g. breast_cancer[dis] or \"breast cancer\"[dis])
        Example: "BRCA1[gene] AND (pathogenic[clinsig] OR likely_pathogenic[clinsig])"


        EXAMPLES OF CORRECT QUERIES:
        - For "pathogenic BRCA1 variants": "BRCA1[gene] AND clinsig_pathogenic[prop]"
        - For "Specific RS": "rs6025[rsid]"
        - For "Combined search with multiple criteria": "BRCA1[gene] AND origin_germline[prop]"
        - For "Find variants in a specific genomic region": "17[chr] AND 43000000:44000000[chrpos37]"
        - If query asks for pathogenicity of a variant, it's asking for all possible germline classifications of the variant, so just [gene] AND [variant] is needed
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=clinvar_schema,
            system_template=system_prompt_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        search_term = query_info.get("search_term", "")

        if not search_term:
            return {
                "error": "Failed to generate a valid search term from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }

    return _query_ncbi_database(
        database="clinvar",
        search_term=search_term,
        max_results=max_results,
    )


def query_geo(
    prompt=None,
    search_term=None,
    max_results=3,
):
    """Query the NCBI Gene Expression Omnibus (GEO) using natural language or a direct search term.

    Parameters
    ----------
    prompt (str, required): Natural language query about RNA-seq, microarray, or other expression data
    search_term (str, optional): Direct search term in GEO syntax
    max_results (int): Maximum number of results to return

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_geo("Find RNA-seq datasets for breast cancer")
    - Direct search: query_geo(search_term="RNA-seq AND breast cancer AND gse[ETYP]")

    """
    if not prompt and not search_term:
        return {"error": "Either a prompt or a search term must be provided"}

    database = "gds"  # Default database

    if prompt:
        # Load GEO schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "geo.pkl")
        with open(schema_path, "rb") as f:
            geo_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a bioinformatics research assistant that helps convert natural language queries into structured GEO (Gene Expression Omnibus) search queries.

        Based on the user's natural language request, you will generate a structured search for the GEO database.

        Output only a JSON object with the following fields:
        1. "search_term": The exact search query to use with the GEO API
        2. "database": The specific GEO database to search (either "gds" for GEO DataSets or "geoprofiles" for GEO Profiles)

        IMPORTANT: Your response must ONLY contain a JSON object with the required fields.

        Your "search_term" MUST strictly follow these GEO search syntax rules/tags:

        {schema}

        For combining terms: Use AND, OR, NOT (must be capitalized)
        For complex logic: Use parentheses
        For terms with multiple words: use double quotes or underscore (e.g. "breast cancer"[Title])
        Date ranges use colon format: 2015/01:2020/12[PDAT]

        Choose the appropriate database based on the user's query:
        - gds: GEO DataSets (contains Series, Datasets, Platforms, Samples metadata)
        - geoprofiles: GEO Profiles (contains gene expression data)

        If database isn't clearly specified, default to "gds" as it contains most common experiment metadata.

        EXAMPLES OF CORRECT OUTPUTS:
        - For "RNA-seq data in breast cancer": {"search_term": "RNA-seq AND breast cancer AND gse[ETYP]", "database": "gds"}
        - For "Mouse microarray data from 2020": {"search_term": "Mus musculus[ORGN] AND 2020[PDAT] AND microarray AND gse[ETYP]", "database": "gds"}
        - For "Expression profiles of TP53 in lung cancer": {"search_term": "TP53[Gene Symbol] AND lung cancer", "database": "geoprofiles"}
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=geo_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the search term and database from Claude's response
        query_info = llm_result["data"]
        search_term = query_info.get("search_term", "")
        database = query_info.get("database", "gds")

        if not search_term:
            return {
                "error": "Failed to generate a valid search term from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }

    # Execute the GEO query using the helper function
    result = _query_ncbi_database(
        database=database,
        search_term=search_term,
        max_results=max_results,
    )

    return result


def query_dbsnp(
    prompt=None,
    search_term=None,
    max_results=3,
):
    """Query the NCBI dbSNP database using natural language or a direct search term.

    Parameters
    ----------
    prompt (str, required): Natural language query about genetic variants/SNPs
    search_term (str, optional): Direct search term in dbSNP syntax
    max_results (int): Maximum number of results to return

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_dbsnp("Find pathogenic variants in BRCA1")
    - Direct search: query_dbsnp(search_term="BRCA1[Gene Name] AND pathogenic[Clinical Significance]")

    """
    if not prompt and not search_term:
        return {"error": "Either a prompt or a search term must be provided"}

    if prompt:
        # Load dbSNP schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "dbsnp.pkl")
        with open(schema_path, "rb") as f:
            dbsnp_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a genetics research assistant that helps convert natural language queries into structured dbSNP search queries.

        Based on the user's natural language request, you will generate a structured search for the dbSNP database.

        Output only a JSON object with the following fields:
        1. "search_term": The exact search query to use with the dbSNP API

        IMPORTANT: Your response must ONLY contain a JSON object with the search term field.

        Your "search_term" MUST strictly follow these dbSNP search syntax rules/tags:

        {schema}

        For combining terms: Use AND, OR, NOT (must be capitalized)
        For complex logic: Use parentheses
        For terms with multiple words: use double quotes (e.g. "breast cancer"[Disease Name])

        EXAMPLES OF CORRECT QUERIES:
        - For "pathogenic variants in BRCA1": "BRCA1[Gene Name] AND pathogenic[Clinical Significance]"
        - For "specific SNP rs6025": "rs6025[rs]"
        - For "SNPs in a genomic region": "17[Chromosome] AND 41196312:41277500[Base Position]"
        - For "common SNPs in EGFR": "EGFR[Gene Name] AND common[COMMON]"
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=dbsnp_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the search term from Claude's response
        query_info = llm_result["data"]
        search_term = query_info.get("search_term", "")

        if not search_term:
            return {
                "error": "Failed to generate a valid search term from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }

    # Execute the dbSNP query using the helper function
    result = _query_ncbi_database(
        database="snp",
        search_term=search_term,
        max_results=max_results,
    )

    return result


def query_ucsc(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the UCSC Genome Browser API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about genomic data
    endpoint (str, optional): Full URL or endpoint specification with parameters
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_ucsc("Get DNA sequence of chromosome M positions 1-100 in human genome")
    - Direct endpoint: query_ucsc(endpoint="https://api.genome.ucsc.edu/getData/sequence?genome=hg38&chrom=chrM&start=1&end=100")

    """
    # Base URL for UCSC API
    base_url = "https://api.genome.ucsc.edu"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load UCSC schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "ucsc.pkl")
        with open(schema_path, "rb") as f:
            ucsc_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a genomics expert specialized in using the UCSC Genome Browser API.

        Based on the user's natural language request, determine the appropriate UCSC Genome Browser API endpoint and parameters.

        UCSC GENOME BROWSER API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://api.genome.ucsc.edu" and all parameters)
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - For chromosome names, always include the "chr" prefix (e.g., "chr1", "chrX", "chrM")
        - Genomic positions are 0-based (first base is position 0)
        - For "start" and "end" parameters, both must be provided together
        - The "maxItemsOutput" parameter can be used to limit the amount of data returned
        - Common genomes include: "hg38" (human), "mm39" (mouse), "danRer11" (zebrafish)
        - For sequence data, use "getData/sequence" endpoint
        - For chromosome listings, use "list/chromosomes" endpoint
        - For available genomes, use "list/ucscGenomes" endpoint

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=ucsc_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }

    else:
        # Process provided endpoint
        if not endpoint.startswith("http"):
            # Add base URL if it's just a path
            endpoint = f"{base_url}/{endpoint}"

        description = "Direct query to UCSC Genome Browser API"

    # Execute the UCSC API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    # Format the results if successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def query_ensembl(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the Ensembl REST API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about genomic data
    endpoint (str, optional): Direct API endpoint to query (e.g., "lookup/symbol/human/BRCA2") or full URL
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_ensembl("Get information about the human BRCA2 gene")
    - Direct endpoint: query_ensembl(endpoint="lookup/symbol/homo_sapiens/BRCA2")

    """
    # Base URL for Ensembl API
    base_url = "https://rest.ensembl.org"

    # Ensure we have either a prompt or an endpoint
    if not prompt and not endpoint:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load Ensembl schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "ensembl.pkl")
        with open(schema_path, "rb") as f:
            ensembl_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a genomics and bioinformatics expert specialized in using the Ensembl REST API.

        Based on the user's natural language request, determine the appropriate Ensembl REST API endpoint and parameters.

        ENSEMBL REST API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "endpoint": The API endpoint to query (e.g., "lookup/symbol/homo_sapiens/BRCA2")
        2. "params": An object containing query parameters specific to the endpoint
        3. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - Chromosome region queries have a maximum length of 4900000 bp inclusive, so bp of start and end should be 4900000 bp apart. If the user's query exceeds this limit, Ensembl will return an error.
        - For symbol lookups, the format is "lookup/symbol/[species]/[symbol]"
        - To find the coordinates of a band on a chromosome, use /info/assembly/homo_sapiens/[chromosome] with parameters "band":1
        - To find the overlapping genes of a genomic region, use /overlap/region/homo_sapiens/[chromosome]:[start]-[end]
        - For sequence queries, specify the sequence type in parameters (genomic, cdna, cds, protein)
        - For converting rsID to hg38 genomic coordinates, use the "GET id/variation/[species]/[rsid]" endpoint
        - Many endpoints support "content-type" parameter for format specification (application/json, text/xml)

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=ensembl_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the endpoint and parameters from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")
        params = query_info.get("params", {})
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        if endpoint.startswith("http"):
            # If a full URL is provided, extract the endpoint part
            if endpoint.startswith(base_url):
                endpoint = endpoint[len(base_url) :].lstrip("/")

        params = {}
        description = "Direct query to Ensembl API"

    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = endpoint[1:]

    # Prepare headers for JSON response
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Construct the URL
    url = f"{base_url}/{endpoint}"

    # Execute the Ensembl API request using the helper function
    api_result = _query_rest_api(
        endpoint=url,
        method="GET",
        params=params,
        headers=headers,
        description=description,
    )

    # Format the results if successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def query_opentarget(
    prompt=None,
    query=None,
    variables=None,
    verbose=False,
):
    """Query the OpenTargets Platform API using natural language or a direct GraphQL query.

    Parameters
    ----------
    prompt (str, required): Natural language query about drug targets, diseases, and mechanisms
    query (str, optional): Direct GraphQL query string
    variables (dict, optional): Variables for the GraphQL query
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_opentarget("Find drug targets for Alzheimer's disease")
    - Direct query: query_opentarget(query="query diseaseAssociations($diseaseId: String!) {...}",
                                     variables={"diseaseId": "EFO_0000249"})

    """
    # Constants and initialization
    OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    # Ensure we have either a prompt or a query
    if prompt is None and query is None:
        return {"error": "Either a prompt or a GraphQL query must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load OpenTargets schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "opentarget.pkl")
        with open(schema_path, "rb") as f:
            opentarget_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are an expert in translating natural language requests into GraphQL queries for the OpenTargets Platform API.

        Here is a schema of the main types and queries available in the OpenTargets Platform API:
        {schema}

        Translate the user's natural language request into a valid GraphQL query for this API.
        Return only a JSON object with two fields:
        1. "query": The complete GraphQL query string
        2. "variables": A JSON object containing the variables needed for the query

        SPECIAL NOTES:
        - Disease IDs typically use EFO ontology (e.g., "EFO_0000249" for Alzheimer's disease)
        - Target IDs typically use Ensembl IDs (e.g., "ENSG00000197386" for ENSG00000197386)
        - The API can provide information about drug-target associations, disease-target associations, etc.
        - Always limit results to a reasonable number using "first" parameter (e.g., first: 10)
        - Always escape special characters, including quotes, in the query string (eg. \\" instead of ")

        Return ONLY the JSON object with no additional text or explanations.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=opentarget_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the query and variables from Claude's response
        query_info = llm_result["data"]
        query = query_info.get("query", "")
        if variables is None:  # Only use Claude's variables if none provided
            variables = query_info.get("variables", {})

        if not query:
            return {
                "error": "Failed to generate a valid GraphQL query from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }

    # Execute the GraphQL query
    api_result = _query_rest_api(
        endpoint=OPENTARGETS_URL,
        method="POST",
        json_data={"query": query, "variables": variables or {}},
        headers={"Content-Type": "application/json"},
        description="OpenTargets Platform GraphQL query",
    )

    # Format the results if not verbose and successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


# Monarch Initiative integration
def query_monarch(
    prompt=None,
    endpoint=None,
    max_results=2,
    verbose=False,
):
    """Query the Monarch Initiative API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, optional): Natural language query about genes, diseases, phenotypes, etc.
    endpoint (str, optional): Direct Monarch API endpoint or full URL
    max_results (int): Maximum number of results to return (if supported by endpoint)
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_monarch("Find phenotypes associated with BRCA1")
    - Direct endpoint: query_monarch(endpoint="https://api.monarchinitiative.org/v3/api/search?q=marfan&category=biolink:Disease&limit=10")
    - Direct endpoint: query_monarch(endpoint="https://api.monarchinitiative.org/v3/api/entity/MONDO:0007947")
    """
    base_url = "https://api.monarchinitiative.org/v3/api"

    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, use Claude to generate the endpoint
    if prompt:
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "monarch.pkl")
        if os.path.exists(schema_path):
            with open(schema_path, "rb") as f:
                monarch_schema = pickle.load(f)
        else:
            monarch_schema = None

        system_template = """
        You are an expert in translating natural language requests into REST API calls for the Monarch Initiative Platform API.

        Here is the API schema with available endpoints and parameters:
        {schema}

        Translate the user's natural language request into a valid REST API call for this API.
        Return only a JSON object with three fields:
        1. "endpoint": The specific endpoint name from the schema
        2. "url": The complete URL with path parameters filled in
        3. "params": A JSON object containing query parameters needed for the request

        SPECIAL NOTES:
        - Disease IDs typically use MONDO ontology (e.g., "MONDO:0007947" for Marfan syndrome)
        - Gene IDs typically use HGNC (e.g., "HGNC:3603" for FBN1) or other standard identifiers
        - Phenotype IDs use Human Phenotype Ontology (e.g., "HP:0002616" for aortic root dilatation)
        - Association categories use biolink model terms (e.g., "biolink:DiseaseToPhenotypicFeatureAssociation")
        - For example: to find phenotypes associated with BRCA1, use the following endpoint: /entity/HGNC:1100/biolink:GeneToPhenotypicFeatureAssociation
        - For search queries, use the 'q' parameter with relevant keywords
        - When looking for associations, use the association_table endpoint with entity ID and category
        - For similarity searches, use semsim endpoints with comma-separated term lists
        - Entity categories include: biolink:Disease, biolink:Gene, biolink:PhenotypicFeature, etc.
        - Format parameter defaults to 'json' but can be 'tsv' for tabular data
        - Use autocomplete endpoint for entity name suggestions before exact searches

        COMMON PATTERNS:
        - Search for entities: Use 'search' endpoint with 'q' and 'category' parameters
        - Get entity details: Use 'get_entity' endpoint with specific ID
        - Find associations: Use 'association_table' endpoint with ID and association category
        - Compare phenotypes: Use 'semsim_compare' with lists of phenotype IDs
        - Find similar diseases: Use 'semsim_search' with phenotype profile

        Return ONLY the JSON object with no additional text or explanations.
        """

        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=monarch_schema,
            system_template=system_template,
        )
        if not llm_result["success"]:
            return llm_result
        query_info = llm_result["data"]
        endpoint = query_info.get("url", "")  # Changed from "full_url" to "url"
        description = f"Monarch API query: {query_info.get('endpoint', 'unknown endpoint')}"
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Use provided endpoint directly
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        description = "Direct query to Monarch API"

    # Add max_results as a query parameter if not already present
    if "?" in endpoint:
        if "rows=" not in endpoint and "limit=" not in endpoint:
            endpoint += f"&limit={max_results}"
    else:
        endpoint += f"?limit={max_results}"

    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


# OpenFDA integration
def query_openfda(
    prompt=None,
    endpoint=None,
    max_results=100,
    verbose=True,
):
    """Query the OpenFDA API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, optional): Natural language query about drugs, adverse events, recalls, etc.
    endpoint (str, optional): Direct OpenFDA API endpoint or full URL
    max_results (int): Maximum number of results to return (if supported by endpoint)
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_openfda("Find adverse events for Lipitor")
    - Direct endpoint: query_openfda(endpoint="https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:lipitor")
    """
    base_url = "https://api.fda.gov"

    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, use Claude or Gemini to generate the endpoint
    if prompt:
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "openfda.pkl")
        if os.path.exists(schema_path):
            with open(schema_path, "rb") as f:
                openfda_schema = pickle.load(f)
        else:
            openfda_schema = None

        system_template = """
        You are a biomedical informatics expert specialized in using the OpenFDA API.\n\nBased on the user's natural language request, determine the appropriate OpenFDA API endpoint and parameters.\n\nOPENFDA API SCHEMA:\n{schema}\n\nYour response should be a JSON object with the following fields:\n1. \"full_url\": The complete URL to query (including the base URL \"https://api.fda.gov\" and any parameters)\n2. \"description\": A brief description of what the query is doing\n\nSPECIAL NOTES:\n- For drug event queries, use /drug/event.json?search=...\n- For drug label queries, use /drug/label.json?search=...\n- For recall queries, use /drug/enforcement.json?search=...\n- Use max_results to limit the number of returned items if supported (limit=)\n- Always URL-encode search terms\n- Return ONLY the JSON object with no additional text.\n        """
        # Select LLM for prompt translation
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=openfda_schema,
            system_template=system_template,
        )
        if not llm_result["success"]:
            return llm_result
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Use provided endpoint directly
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"

    # Add max_results as a query parameter if not already present
    if "?" in endpoint:
        if "limit=" not in endpoint:
            endpoint += f"&limit={max_results}"
    else:
        endpoint += f"?limit={max_results}"


def query_clinicaltrials(
    prompt: str | None = None,
    endpoint: str | None = None,
    term: str | None = None,
    status: str | None = None,
    condition: str | None = None,
    intervention: str | None = None,
    location: str | None = None,
    phase: str | None = None,
    page_size: int = 10,
    max_pages: int = 1,
    page_token: str | None = None,
    verbose: bool = True,
):
    """Query ClinicalTrials.gov API for clinical studies.

    Modes:
    - Direct URL: set `endpoint` to a full URL or a path (e.g., "/studies?query.term=breast%20cancer").
    - Structured params: provide `term`, `status`, `condition`, etc.
    - Natural language: provide `prompt` and the function will infer structured params.

    Returns a dict with aggregated results across pages (up to `max_pages`).
    """
    base_url = "https://clinicaltrials.gov/api/v2"

    # If natural language prompt is provided, ask LLM to produce parameters
    if prompt and not endpoint and not term:
        system_template = (
            "You translate natural language into ClinicalTrials.gov API parameters.\n"
            "Return ONLY a JSON object with keys among: \n"
            '{"term": str, "status": str, "condition": str, "intervention": str, "location": str, "phase": str, "page_size": int}.\n'
            "Do not include explanations. Keep values concise (e.g., status like 'RECRUITING', phase like 'PHASE3')."
        )

        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=None,
            system_template=system_template,
        )
        if llm_result.get("success"):
            mapping = llm_result["data"] or {}
            term = mapping.get("term", term)
            status = mapping.get("status", status)
            condition = mapping.get("condition", condition)
            intervention = mapping.get("intervention", intervention)
            location = mapping.get("location", location)
            phase = mapping.get("phase", phase)
            page_size = int(mapping.get("page_size", page_size) or page_size)

    # If endpoint provided, normalize to full URL
    description = "ClinicalTrials.gov studies query"
    if endpoint:
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"

        api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)
        if not verbose and api_result.get("success") and "result" in api_result:
            return _format_query_results(api_result["result"])
        return api_result

    # Otherwise build params for /studies
    if not term and not condition and not intervention and not status:
        return {"error": "Provide at least one of: prompt, term, condition, intervention, status, or a direct endpoint"}

    url = f"{base_url}/studies"

    def build_params(token: str | None) -> dict[str, Any]:
        params: dict[str, Any] = {"pageSize": max(1, min(int(page_size), 100))}
        if term:
            params["query.term"] = term
        if status:
            params["filter.overallStatus"] = status
        if condition:
            params["filter.condition"] = condition
        if intervention:
            params["filter.intervention"] = intervention
        if location:
            params["filter.location"] = location
        if phase:
            params["filter.phase"] = phase
        if token:
            params["pageToken"] = token
        return params

    aggregated = {
        "success": True,
        "query_info": {
            "endpoint": url,
            "description": description,
            "parameters": {
                "term": term,
                "status": status,
                "condition": condition,
                "intervention": intervention,
                "location": location,
                "phase": phase,
                "page_size": page_size,
                "max_pages": max_pages,
            },
        },
        "result": {"studies": [], "page_count": 0},
    }

    current_token = page_token
    pages_fetched = 0
    while pages_fetched < max_pages:
        params = build_params(current_token)
        page_resp = _query_rest_api(endpoint=url, method="GET", params=params, description=description)
        if not page_resp.get("success"):
            return page_resp

        data = page_resp.get("result") or {}
        studies = data.get("studies") or data.get("items") or []
        aggregated["result"]["studies"].extend(studies)
        pages_fetched += 1
        aggregated["result"]["page_count"] = pages_fetched

        # Continue if next token present
        current_token = data.get("nextPageToken") or data.get("nextPage") or None
        if not current_token:
            break

    if not verbose:
        return _format_query_results(aggregated)
    return aggregated

    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def query_gwas_catalog(
    prompt=None,
    endpoint=None,
    max_results=3,
):
    """Query the GWAS Catalog API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about GWAS data
    endpoint (str, optional): Full API endpoint to query (e.g., "https://www.ebi.ac.uk/gwas/rest/api/studies?diseaseTraitId=EFO_0001360")
    max_results (int): Maximum number of results to return

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_gwas_catalog("Find GWAS studies related to Type 2 diabetes")
    - Direct endpoint: query_gwas_catalog(endpoint="studies", params={"diseaseTraitId": "EFO_0001360"})

    """
    # Base URL for GWAS Catalog API
    base_url = "https://www.ebi.ac.uk/gwas/rest/api"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load GWAS Catalog schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "gwas_catalog.pkl")
        with open(schema_path, "rb") as f:
            gwas_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a genomics expert specialized in using the GWAS Catalog API.

        Based on the user's natural language request, determine the appropriate GWAS Catalog API endpoint and parameters.

        GWAS CATALOG API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "endpoint": The API endpoint to query (e.g., "studies", "associations")
        2. "params": An object containing query parameters specific to the endpoint
        3. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - For disease/trait searches, consider using the "EFO" identifiers when possible
        - Common endpoints include: "studies", "associations", "singleNucleotidePolymorphisms", "efoTraits"
        - For pagination, use "size" and "page" parameters
        - For filtering by p-value, use "pvalueMax" parameter
        - GWAS Catalog uses a HAL-based REST API

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=gwas_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the endpoint and parameters from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")
        params = query_info.get("params", {})
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        if endpoint is None:
            endpoint = ""  # Use root endpoint
        params = {"size": max_results}
        description = f"Direct query to {endpoint}"

    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = endpoint[1:]

    # Construct the URL
    url = f"{base_url}/{endpoint}"

    # Execute the GWAS Catalog API request using the helper function
    api_result = _query_rest_api(endpoint=url, method="GET", params=params, description=description)

    return api_result


def query_gnomad(
    prompt=None,
    gene_symbol=None,
    verbose=True,
):
    """Query gnomAD for variants in a gene using natural language or direct gene symbol.

    Parameters
    ----------
    prompt (str, required): Natural language query about genetic variants
    gene_symbol (str, optional): Gene symbol (e.g., "BRCA1")
    verbose (bool): Whether to print verbose output

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Direct gene: query_gnomad(gene_symbol="BRCA1")
    - Natural language: query_gnomad(prompt="Find variants in the TP53 gene")

    """
    # Base URL for gnomAD API
    base_url = "https://gnomad.broadinstitute.org/api"

    # Ensure we have either a prompt or a gene_symbol
    if prompt is None and gene_symbol is None:
        return {"error": "Either a prompt or a gene_symbol must be provided"}

    # If using prompt, parse with Claude
    if prompt and not gene_symbol:
        # Load gnomAD schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "gnomad.pkl")
        with open(schema_path, "rb") as f:
            gnomad_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a genomics expert specialized in using the gnomAD GraphQL API.

        Based on the user's natural language request, extract the gene symbol and relevant parameters and create the gnomAD GraphQL query.

        GnomAD GraphQL API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "query": The complete GraphQL query string

        SPECIAL NOTES:
        - The gene_symbol should be the official gene symbol (e.g., "BRCA1" not "breast cancer gene 1")
        - If no reference genome is specified, default to GRCh38
        - If no dataset is specified, default to gnomad_r4
        - Return only a single gene symbol, even if multiple are mentioned
        - Always escape special characters, including quotes, in the query string (eg. \" instead of ")



        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=gnomad_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the gene symbol from Claude's response
        query_info = llm_result["data"]
        query_str = query_info.get("query", "")

        if not query_str:
            return {
                "error": "Failed to extract a valid query from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        description = f"Query gnomAD for variants in {gene_symbol}"
        # replace BRCA1 with gene_symbol
        query_str = gnomad_schema.replace("BRCA1", gene_symbol)

    api_result = _query_rest_api(
        endpoint=base_url,
        method="POST",
        json_data={"query": query_str},
        headers={"Content-Type": "application/json"},
        description=description,
    )

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def blast_sequence(sequence: str, database: str, program: str) -> dict[str, str | float] | str:
    """Identifies a DNA sequence using NCBI BLAST with improved error handling, timeout management, and debugging.

    Args:
        sequence (str): The sequence to identify. If DNA, use database: core_nt, program: blastn;
                        if protein, use database: nr, program: blastp
        database (str): The BLAST database to search against
        program (str): The BLAST program to use

    Returns:
        dict: A dictionary containing the title, e-value, identity percentage, and coverage percentage of the best alignment

    """
    max_attempts = 1  # One initial attempt plus one retry
    attempts = 0
    max_runtime = 600  # 10 minutes in seconds

    while attempts < max_attempts:
        try:
            attempts += 1
            query_sequence = Seq(sequence)

            # Start timer
            start_time = time.time()

            # Submit BLAST job
            print(f"Submitting BLAST job (attempt {attempts}/{max_attempts})...")
            result_handle = NCBIWWW.qblast(
                program,
                database,
                query_sequence,
                expect=100,
                word_size=7,
                megablast=True,
            )

            # Parse results with timeout check
            blast_records = NCBIXML.parse(result_handle)
            blast_record = None

            # Try to get the first record with timeout check
            while time.time() - start_time < max_runtime:
                try:
                    # Set a short timeout for next operation
                    blast_record = next(blast_records)  # Get first record
                    break  # Successfully got the record
                except StopIteration:
                    # No more records
                    return "No BLAST results found"
                except Exception:
                    # Check if we've exceeded the time limit
                    if time.time() - start_time >= max_runtime:
                        if attempts < max_attempts:
                            print("BLAST job timeout exceeded. Resubmitting...")
                            break  # Break to retry
                        else:
                            return "BLAST search failed after maximum attempts due to timeout"
                    # Brief pause before trying again
                    time.sleep(1)

            # Check if we timed out during record retrieval
            if blast_record is None:
                if attempts < max_attempts:
                    continue  # Retry
                else:
                    return "BLAST search failed after maximum attempts due to timeout"

            # Debug information
            print(f"Number of alignments found: {len(blast_record.alignments)}")

            if blast_record.alignments:
                for alignment in blast_record.alignments:
                    print("\nAlignment:")
                    print(f"hit_id: {alignment.hit_id}")
                    print(f"hit_def: {alignment.hit_def}")
                    print(f"accession: {alignment.accession}")
                    for hsp in alignment.hsps:
                        print(f"E-value: {hsp.expect}")
                        print(f"Score: {hsp.score}")
                        print(f"Identities: {hsp.identities}/{hsp.align_length}")

                        return {
                            "hit_id": alignment.hit_id,
                            "hit_def": alignment.hit_def,
                            "accession": alignment.accession,
                            "e_value": hsp.expect,
                            "identity": (hsp.identities / float(hsp.align_length)) * 100,
                            "coverage": len(hsp.query) / len(sequence) * 100,
                        }
            else:
                return "No alignments found - sequence might be too short or low complexity"

        except Exception as e:
            if attempts < max_attempts:
                print(f"Error during BLAST search: {str(e)}. Retrying...")
                time.sleep(2)  # Wait briefly before retrying
            else:
                return f"Error during BLAST search after maximum attempts: {str(e)}"

    return "BLAST search failed after maximum attempts"


def query_reactome(
    prompt=None,
    endpoint=None,
    download=False,
    output_dir=None,
    verbose=True,
):
    """Query the Reactome database using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about biological pathways
    endpoint (str, optional): Direct API endpoint or full URL
    download (bool): Whether to download pathway diagrams
    output_dir (str, optional): Directory to save downloaded files
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_reactome("Find pathways related to DNA repair")
    - Direct endpoint: query_reactome(endpoint="data/pathways/R-HSA-73894")

    """
    # Base URLs for Reactome APIs
    content_base_url = "https://reactome.org/ContentService"
    analysis_base_url = "https://reactome.org/AnalysisService"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # Create output directory if downloading and directory doesn't exist
    if download and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # If using prompt, parse with Claude
    if prompt:
        # Load Reactome schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "reactome.pkl")
        with open(schema_path, "rb") as f:
            reactome_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a bioinformatics expert specialized in using the Reactome API.

        Based on the user's natural language request, determine the appropriate Reactome API endpoint and parameters.

        REACTOME API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "endpoint": The API endpoint to query (e.g., "data/pathways/PATHWAY_ID", "data/query/GENE_SYMBOL")
        2. "base": Which base URL to use ("content" for ContentService or "analysis" for AnalysisService)
        3. "params": An object containing query parameters specific to the endpoint
        4. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - Reactome has two primary APIs: ContentService (for retrieving specific pathway data) and AnalysisService (for analyzing gene lists)
        - For pathway queries, use "data/pathways/PATHWAY_ID" with the pathway stable identifier (e.g., R-HSA-73894)
        - For gene queries, use "data/query/GENE" with official gene symbol (e.g., "BRCA1")
        - For pathway diagrams, include "download: true" in your response if the query is for pathway visualization
        - Common human pathway IDs start with "R-HSA-"

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=reactome_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the endpoint and parameters from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")
        base = query_info.get("base", "content")  # Default to ContentService
        params = query_info.get("params", {})
        description = query_info.get("description", "")
        should_download = query_info.get("download", download)  # Override download if specified

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        if endpoint.startswith("http"):
            # Full URL already provided
            if "ContentService" in endpoint:
                base = "content"
            elif "AnalysisService" in endpoint:
                base = "analysis"
            else:
                base = "content"  # Default
        else:
            # Just endpoint provided, assume ContentService by default
            base = "content"

        params = {}
        description = f"Direct query to Reactome {base} API: {endpoint}"
        should_download = download

    # Select base URL based on API type
    base_url = content_base_url if base == "content" else analysis_base_url

    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = endpoint[1:]

    # Construct the URL
    if endpoint.startswith("http"):
        url = endpoint  # Full URL already provided
    else:
        url = f"{base_url}/{endpoint}"

    # Execute the Reactome API request using the helper function
    api_result = _query_rest_api(endpoint=url, method="GET", params=params, description=description)

    # Handle downloading pathway diagrams if requested
    if should_download and api_result.get("success") and "result" in api_result:
        result = api_result["result"]
        pathway_id = None

        # Try to extract pathway ID from result
        if isinstance(result, dict):
            pathway_id = result.get("stId") or result.get("dbId")

        # If we have a pathway ID and output directory, download diagram
        if pathway_id and output_dir:
            diagram_url = f"{content_base_url}/data/pathway/{pathway_id}/diagram"
            try:
                diagram_response = requests.get(diagram_url)
                diagram_response.raise_for_status()

                # Save diagram file
                diagram_path = os.path.join(output_dir, f"{pathway_id}_diagram.png")
                with open(diagram_path, "wb") as f:
                    f.write(diagram_response.content)

                api_result["diagram_path"] = diagram_path
            except Exception as e:
                api_result["diagram_error"] = f"Failed to download diagram: {str(e)}"

    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


def query_regulomedb(
    prompt=None,
    endpoint=None,
    verbose=False,
):
    """Query the RegulomeDB database using natural language or direct variant/coordinate specification.

    Parameters
    ----------
    prompt (str, required): Natural language query about regulatory elements
    endpoint (str, optional): The full endpoint to query (e.g., "https://regulomedb.org/regulome-search/?regions=chr11:5246919-5246919&genome=GRCh38")
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_regulomedb("Find regulatory elements for rs35675666")
    - Direct variant: query_regulomedb(variant="rs35675666")
    - Coordinates: query_regulomedb(coordinates="chr11:5246919-5246919")

    """
    # Base URL for RegulomeDB API

    # Ensure we have either a prompt, variant, or coordinates
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt, variant ID, or genomic coordinates must be provided"}

    # If using prompt, parse with Claude
    if prompt and not endpoint:
        # Create system prompt template
        system_template = """
        You are a genomics expert specialized in using the RegulomeDB API.

        Based on the user's natural language request, extract the variant ID or genomic coordinates they want to query.

        Your response should be a JSON object with ONLY ONE of the following fields:
        1. "endpoint": The API endpoint to query (e.g., "https://regulomedb.org/regulome-search/?regions=chr11:5246919-5246919&genome=GRCh38")


        SPECIAL NOTES:
        - RegulomeDB only works with human genome data
        - Variant IDs should be rsIDs from dbSNP when possible. The endpoint should be in the format https://regulomedb.org/regulome-search/?regions=rsID&genome=GRCh38
        - Thumbnails for chip and chromatin should be in the format https://regulomedb.org/regulome-search?regions=chr11:5246919-5246919&genome=GRCh38/thumbnail=chip
        - Coordinates should be in GRCh37/hg19 format
        - For single base queries, use the same position for start and end (e.g., "chr11:5246919-5246919")
        - Chromosome should be specified with "chr" prefix (e.g., "chr11" not just "11")

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=None,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the variant or coordinates from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")

        if not endpoint:
            return {
                "error": "Failed to extract a valid variant ID or coordinates from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        pass

    # Construct the request URL
    endpoint = endpoint

    # Execute the RegulomeDB API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", headers={"Accept": "application/json"})

    # Format the results if not verbose and successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_pride(
    prompt=None,
    endpoint=None,
    max_results=3,
):
    """Query the PRIDE (PRoteomics IDEntifications) database using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about proteomics data
    endpoint (str, optional): The full endpoint to query (e.g., "https://www.ebi.ac.uk/pride/ws/archive/v2/projects?keyword=breast%20cancer")
    max_results (int): Maximum number of results to return

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_pride("Find proteomics data related to breast cancer")
    - Direct endpoint: query_pride(endpoint="projects", params={"keyword": "breast cancer"})

    """
    # Base URL for PRIDE API
    base_url = "https://www.ebi.ac.uk/pride/ws/archive/v2"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load PRIDE schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "pride.pkl")
        with open(schema_path, "rb") as f:
            pride_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a proteomics expert specialized in using the PRIDE API.

        Based on the user's natural language request, determine the appropriate PRIDE API endpoint and parameters.

        PRIDE API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "endpoint": The full url endpoint to query
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - PRIDE is a repository for proteomics data stored at EBI
        - Common endpoints include: "projects", "assays", "files", "proteins", "peptideevidences"
        - For searching projects, you can use parameters like "keyword", "species", "tissue", "disease"
        - For pagination, use "page" and "pageSize" parameters
        - Most results include PagingObject and FieldsObject structures

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=pride_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the endpoint and parameters from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")
        params = query_info.get("params", {})
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        params = {"pageSize": max_results, "page": 0}
        description = f"Direct query to PRIDE {endpoint}"

    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = f"{base_url}{endpoint}"
    elif not endpoint.startswith("http"):
        endpoint = f"{base_url}/{endpoint.lstrip('/')}"
    description = "Direct query to provided endpoint"

    # Execute the PRIDE API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", params=params, description=description)

    return api_result


def query_gtopdb(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the Guide to PHARMACOLOGY database (GtoPdb) using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about drug targets, ligands, and interactions
    endpoint (str, optional): Full API endpoint to query (e.g., "https://www.guidetopharmacology.org/services/targets?type=GPCR&name=beta-2")
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_gtopdb("Find ligands that target the beta-2 adrenergic receptor")
    - Direct endpoint: query_gtopdb(endpoint="targets", params={"type": "GPCR", "name": "beta-2"})

    """
    # Base URL for GtoPdb API
    base_url = "https://www.guidetopharmacology.org/services"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load GtoPdb schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "gtopdb.pkl")
        with open(schema_path, "rb") as f:
            gtopdb_schema = pickle.load(f)

        # Create system prompt template
        system_template = r"""
        You are a pharmacology expert specialized in using the Guide to PHARMACOLOGY API.

        Based on the user's natural language request, determine the appropriate GtoPdb API endpoint and parameters.

        GTOPDB API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "endpoint": The full API endpoint to query
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - Main endpoints include: "targets", "ligands", "interactions", "diseases", "refs"
        - Target types include: "GPCR", "NHR", "LGIC", "VGIC", "OtherIC", "Enzyme", "CatalyticReceptor", "Transporter", "OtherProtein"
        - Ligand types include: "Synthetic organic", "Metabolite", "Natural product", "Endogenous peptide", "Peptide", "Antibody", "Inorganic", "Approved", "Withdrawn", "Labelled", "INN"
        - Interaction types include: "Activator", "Agonist", "Allosteric modulator", "Antagonist", "Antibody", "Channel blocker", "Gating inhibitor", "Inhibitor", "Subunit-specific"
        - For specific target/ligand details, use formats like "targets/\{targetId\}" or "ligands/\{ligandId\}"
        - For subresources, use formats like "targets/\{targetId\}/interactions" or "ligands/\{ligandId\}/structure"

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=gtopdb_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the endpoint and parameters from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        description = f"Direct query to GtoPdb {endpoint}"

    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = f"{base_url}{endpoint}"
    elif not endpoint.startswith("http"):
        endpoint = f"{base_url}/{endpoint.lstrip('/')}"
    description = "Direct query to provided endpoint"

    # Execute the GtoPdb API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    # Format the results if not verbose and successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def region_to_ccre_screen(coord_chrom: str, coord_start: int, coord_end: int, assembly: str = "GRCh38") -> str:
    """Given starting and ending coordinates, this function retrieves information of intersecting cCREs.

    Args:
        assembly (str): Assembly of the genome, formatted like 'GRCh38'. Default is 'GRCh38'.
        coord_chrom (str): Chromosome of the gene, formatted like 'chr12'.
        coord_start (int): Starting chromosome coordinate.
        coord_end (int): Ending chromosome coordinate.

    Returns:
        str: A detailed string explaining the steps and the intersecting cCRE data or any error encountered.

    """
    steps = []
    try:
        steps.append(
            f"Starting cCRE data retrieval for coordinates: {coord_chrom}:{coord_start}-{coord_end} (Assembly: {assembly})."
        )

        # Build the URL and request payload
        url = "https://screen-beta-api.wenglab.org/dataws/cre_table"
        data = {
            "assembly": assembly,
            "coord_chrom": coord_chrom,
            "coord_start": coord_start,
            "coord_end": coord_end,
        }

        steps.append("Sending POST request to API with the following data:")
        steps.append(str(data))

        # Make the request
        response = requests.post(url, json=data)

        # Check if the response is successful
        if not response.ok:
            raise Exception(f"Request failed with status code {response.status_code}. Response: {response.text}")

        steps.append("Request executed successfully. Parsing the response...")

        # Parse the JSON response
        response_json = response.json()
        if "errors" in response_json:
            raise Exception(f"API error: {response_json['errors']}")

        # Function to reduce and filter response data
        def reduce_tokens(res_json):
            # Remove unnecessary fields and round floats
            res = sorted(res_json["cres"], key=lambda x: x["dnase_zscore"], reverse=True)
            filtered_res = []

            for item in res:
                new_item = {
                    "chrom": item["chrom"],
                    "start": item["start"],
                    "len": item["len"],
                    "pct": item["pct"],
                    "ctcf_zscore": round(item["ctcf_zscore"], 2),
                    "dnase_zscore": round(item["dnase_zscore"], 2),
                    "enhancer_zscore": round(item["enhancer_zscore"], 2),
                    "promoter_zscore": round(item["promoter_zscore"], 2),
                    "accession": item["info"]["accession"],
                    "isproximal": item["info"]["isproximal"],
                    "concordance": item["info"]["concordant"],
                    "ctcfmax": round(item["info"]["ctcfmax"], 2),
                    "k4me3max": round(item["info"]["k4me3max"], 2),
                    "k27acmax": round(item["info"]["k27acmax"], 2),
                }
                filtered_res.append(new_item)
            return filtered_res

        # Process the response data
        filtered_data = reduce_tokens(response_json)

        if not filtered_data:
            steps.append(f"No intersecting cCREs found for coordinates: {coord_chrom}:{coord_start}-{coord_end}.")
            return "\n".join(steps + ["No cCRE data available for this genomic region."])

        # Format the result into a readable string
        ccre_data_string = f"Intersecting cCREs for {coord_chrom}:{coord_start}-{coord_end} (Assembly: {assembly}):\n"
        for i, ccre in enumerate(filtered_data, 1):
            ccre_data_string += (
                f"cCRE {i}:\n"
                f"  Chromosome: {ccre['chrom']}\n"
                f"  Start: {ccre['start']}\n"
                f"  Length: {ccre['len']}\n"
                f"  PCT: {ccre['pct']}\n"
                f"  CTCF Z-score: {ccre['ctcf_zscore']}\n"
                f"  DNase Z-score: {ccre['dnase_zscore']}\n"
                f"  Enhancer Z-score: {ccre['enhancer_zscore']}\n"
                f"  Promoter Z-score: {ccre['promoter_zscore']}\n"
                f"  Accession: {ccre['accession']}\n"
                f"  Is Proximal: {ccre['isproximal']}\n"
                f"  Concordance: {ccre['concordance']}\n"
                f"  CTCFmax: {ccre['ctcfmax']}\n"
                f"  K4me3max: {ccre['k4me3max']}\n"
                f"  K27acmax: {ccre['k27acmax']}\n\n"
            )

        steps.append(f"cCRE data successfully retrieved and formatted for {coord_chrom}:{coord_start}-{coord_end}.")
        return "\n".join(steps + [ccre_data_string])

    except Exception as e:
        steps.append(f"Exception encountered: {str(e)}")
        return "\n".join(steps + [f"Error: {str(e)}"])


def get_genes_near_ccre(accession: str, assembly: str, chromosome: str, k: int = 10) -> str:
    """Given a cCRE (Candidate cis-Regulatory Element), this function returns a string containing the
    steps it performs and the k nearest genes sorted by distance.

    Parameters
    ----------
    - accession (str): ENCODE Accession ID of query cCRE, e.g., EH38E1516980.
    - assembly (str): Assembly of the gene, e.g., 'GRCh38'.
    - chromosome (str): Chromosome of the gene, e.g., 'chr12'.
    - k (int): Number of nearby genes to return, sorted by distance. Default is 10.

    Returns
    -------
    - str: Steps performed and the result.

    """
    steps_log = (
        f"Starting process with accession: {accession}, assembly: {assembly}, chromosome: {chromosome}, k: {k}\n"
    )

    url = "https://screen-beta-api.wenglab.org/dataws/re_detail/nearbyGenomic"
    data = {"accession": accession, "assembly": assembly, "coord_chrom": chromosome}

    steps_log += "Sending POST request to API with given data.\n"
    response = requests.post(url, json=data)

    if not response.ok:
        steps_log += f"API request failed with response: {response.text}\n"
        return steps_log

    response_json = response.json()

    if "errors" in response_json:
        steps_log += f"API returned errors: {response_json['errors']}\n"
        return steps_log

    nearby_genes = response_json.get(accession, {}).get("nearby_genes", [])
    if not nearby_genes:
        steps_log += "No nearby genes found for the given accession.\n"
        return steps_log

    steps_log += "Successfully retrieved nearby genes. Sorting them by distance.\n"
    sorted_genes = sorted(nearby_genes, key=lambda x: x["distance"])[:k]

    steps_log += f"Returning the top {k} nearest genes.\n"
    steps_log += "Result:\n"

    for gene in sorted_genes:
        gene_name = gene.get("name", "Unknown")
        distance = gene.get("distance", "N/A")
        ensembl_id = gene.get("ensemblid_ver", "N/A")
        start = gene.get("start", "N/A")
        stop = gene.get("stop", "N/A")
        chrom = gene.get("chrom", "N/A")
        steps_log += f"Gene: {gene_name}, Distance: {distance}, Ensembl ID: {ensembl_id}, Chromosome: {chrom}, Start: {start}, Stop: {stop}\n"

    return steps_log


def query_remap(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the ReMap database for regulatory elements and transcription factor binding sites.

    Parameters
    ----------
    prompt (str, required): Natural language query about transcription factors and binding sites
    endpoint (str, optional): Full API endpoint to query (e.g., "https://remap.univ-amu.fr/api/v1/catalogue/tf?tf=CTCF")
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_remap("Find CTCF binding sites in chromosome 1")
    - Direct endpoint: query_remap(endpoint="catalogue/tf", params={"tf": "CTCF"})

    """
    # Base URL for ReMap API
    base_url = "https://remap.univ-amu.fr/api/v1"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load ReMap schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "remap.pkl")
        with open(schema_path, "rb") as f:
            remap_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a genomics expert specialized in using the ReMap database API.

        Based on the user's natural language request, determine the appropriate ReMap API endpoint and parameters.

        REMAP API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "endpoint": The full url endpoint to query
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - ReMap is a database of regulatory regions and transcription factor binding sites based on ChIP-seq experiments
        - Common endpoints include: "catalogue/tf" (transcription factors), "catalogue/biotype" (biotypes), "browse/peaks" (binding sites)
        - For searching binding sites, you can filter by transcription factor (tf), cell line, biotype, chromosome, etc.
        - Genomic coordinates should be specified with "chr", "start", and "end" parameters
        - For limiting results, use "limit" parameter (default is 100)

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=remap_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the endpoint and parameters from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        description = f"Direct query to ReMap {endpoint}"

    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = f"{base_url}{endpoint}"
    elif not endpoint.startswith("http"):
        endpoint = f"{base_url}/{endpoint.lstrip('/')}"
    description = "Direct query to provided endpoint"

    # Execute the ReMap API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    # Format the results if not verbose and successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_mpd(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the Mouse Phenome Database (MPD) for mouse strain phenotype data.

    Parameters
    ----------
    prompt (str, required): Natural language query about mouse phenotypes, strains, or measurements
    endpoint (str, optional): Full API endpoint to query (e.g., "https://phenomedoc.jax.org/MPD_API/strains")
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_mpd("Find phenotype data for C57BL/6J mice related to blood glucose")
    - Direct endpoint: query_mpd(endpoint="strains/C57BL/6J/measures")

    """
    # Base URL for MPD API
    base_url = "https://phenome.jax.org"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load MPD schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "mpd.pkl")
        with open(schema_path, "rb") as f:
            mpd_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a mouse genetics expert specialized in using the Mouse Phenome Database (MPD) API.

        Based on the user's natural language request, determine the appropriate MPD API endpoint and parameters.

        MPD API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "endpoint": The full url endpoint to query (e.g. https://phenome.jax.org/api/strains)
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - The MPD contains phenotype data for diverse strains of laboratory mice
        - Common endpoints include: "strains" (mouse strains), "measures" (phenotypic measurements), "genes" (gene info)
        - Use the url to construct the endpoint, not the endpoint name
        - Common mouse strains include: "C57BL/6J", "DBA/2J", "BALB/cJ", "A/J", "129S1/SvImJ"
        - Common phenotypic domains include: "behavior", "blood_chemistry", "body_weight", "cardiovascular", "growth", "metabolism"

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=mpd_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the endpoint and parameters from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        description = f"Direct query to MPD {endpoint}"

    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = f"{base_url}{endpoint}"
    elif not endpoint.startswith("http"):
        endpoint = f"{base_url}/{endpoint.lstrip('/')}"
    description = "Direct query to provided endpoint"

    # Execute the MPD API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", description=description)

    # Format the results if not verbose and successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result


def query_emdb(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the Electron Microscopy Data Bank (EMDB) for 3D macromolecular structures.

    Parameters
    ----------
    prompt (str, required): Natural language query about EM structures and associated data
    endpoint (str, optional): Full API endpoint to query (e.g., "https://www.ebi.ac.uk/emdb/api/search")
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_emdb("Find cryo-EM structures of ribosomes at resolution better than 3Å")
    - Direct endpoint: query_emdb(endpoint="entry/EMD-10000")

    """
    # Base URL for EMDB API
    base_url = "https://www.ebi.ac.uk/emdb/api"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load EMDB schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "emdb.pkl")
        with open(schema_path, "rb") as f:
            emdb_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a structural biology expert specialized in using the Electron Microscopy Data Bank (EMDB) API.

        Based on the user's natural language request, determine the appropriate EMDB API endpoint and parameters.

        EMDB API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "endpoint": The API endpoint to query (e.g., "search", "entry/EMD-XXXXX")
        2. "params": An object containing query parameters specific to the endpoint
        3. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - EMDB contains 3D macromolecular structures determined by electron microscopy
        - Common endpoints include: "search" (search for entries), "entry/EMD-XXXXX" (specific entry details)
        - For searching, you can filter by resolution, specimen, authors, release date, etc.
        - Resolution filters should be specified with "resolution_low" and "resolution_high" parameters
        - For specific entry retrieval, use the format "entry/EMD-XXXXX" where XXXXX is the EMDB ID
        - Common specimen types include: "ribosome", "virus", "membrane protein", "filament"

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=emdb_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the endpoint and parameters from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")
        params = query_info.get("params", {})
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        params = {}
        description = f"Direct query to EMDB {endpoint}"

    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = f"{base_url}{endpoint}"
    elif not endpoint.startswith("http"):
        endpoint = f"{base_url}/{endpoint.lstrip('/')}"
    description = "Direct query to provided endpoint"

    # Execute the EMDB API request using the helper function
    api_result = _query_rest_api(endpoint=endpoint, method="GET", params=params, description=description)

    # Format the results if not verbose and successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        api_result["result"] = _format_query_results(api_result["result"])

    return api_result
