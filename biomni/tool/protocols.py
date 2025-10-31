"""
Protocols.io API integration for searching and retrieving biological protocols.

This module provides functions to search protocols.io for relevant protocols
based on natural language queries.
"""

import os
from typing import Any

import requests

try:
    # Optional import to read from central config if available
    from biomni.config import default_config  # type: ignore
except Exception:
    default_config = None  # fallback if import fails


# API Configuration
PROTOCOLS_IO_API_BASE = "https://www.protocols.io/api/v3"

# Resolve access token from env or config (no hardcoded defaults)
ACCESS_TOKEN = os.getenv("PROTOCOLS_IO_ACCESS_TOKEN") or os.getenv("BIOMNI_PROTOCOLS_IO_ACCESS_TOKEN")
if not ACCESS_TOKEN and default_config is not None:
    ACCESS_TOKEN = getattr(default_config, "protocols_io_access_token", None)


def search_protocols(
    query: str,
) -> dict[str, Any]:
    """
    Search protocols.io for relevant protocols based on a natural language query.

    Args:
        query: Keyword search term to find protocols (searches names, descriptions, authors), should be EXACT as we are looking for EXACT MATCH

    Returns:
        Dictionary containing:
            - protocols: List of protocol dictionaries with title, description, url, etc.
            - pagination: Pagination information
            - total_results: Total number of matching protocols
            - status_code: API status code (0 = success)

    Raises:
        requests.RequestException: If API request fails
        ValueError: If invalid parameters are provided

    Example:
        >>> results = search_protocols("CRISPR gene editing", page_size=5)
        >>> for protocol in results["protocols"]:
        ...     print(f"{protocol['title']}: {protocol['url']}")
    """
    # Validate parameters
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    # Ensure access token is configured
    if not ACCESS_TOKEN:
        raise ValueError(
            "Protocols.io access token is not configured. Set PROTOCOLS_IO_ACCESS_TOKEN or BIOMNI_PROTOCOLS_IO_ACCESS_TOKEN env var, or configure BiomniConfig.protocols_io_access_token."
        )

    # Construct API request
    url = "https://www.protocols.io/api/v3/protocols"

    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}

    # Prepare search query - wrap in quotes for exact match if requested
    search_key = query.strip()

    params = {
        "filter": "public",  # Search public protocols
        "key": search_key,
    }

    try:
        # Make API request
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()

        # Parse response
        if data.get("status_code") != 0:
            return {
                "protocols": [],
                "pagination": {},
                "total_results": 0,
                "status_code": data.get("status_code"),
                "error": data.get("error_message", "Unknown error"),
            }

        # Extract protocol information
        protocols = []
        for item in data.get("items", []):
            protocol_info = {
                "id": item.get("id"),
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "url": item.get("uri", ""),
                "doi": item.get("doi", ""),
                "authors": [
                    {"name": author.get("name", ""), "username": author.get("username", "")}
                    for author in item.get("authors", [])
                ],
                "created_on": item.get("created_on"),
                "published_on": item.get("published_on"),
                "version": item.get("version_id"),
                "views": item.get("nr_of_views", 0),
                "is_peer_reviewed": item.get("is_peer_reviewed", False),
                "tags": item.get("tags", []),
            }
            protocols.append(protocol_info)

        # Extract pagination info
        pagination = data.get("pagination", {})

        return {
            "protocols": protocols,
            "pagination": {
                "current_page": pagination.get("current_page", 1),
                "total_pages": pagination.get("total_pages", 1),
                "page_size": pagination.get("page_size", 1),
                "total_results": pagination.get("total_results", 0),
            },
            "total_results": pagination.get("total_results", 0),
            "status_code": 0,
        }

    except requests.exceptions.RequestException as e:
        raise requests.RequestException(f"Failed to search protocols.io: {str(e)}") from e


def get_protocol_details(protocol_id: int, timeout: int = 30) -> dict[str, Any]:
    """
    Get detailed information about a specific protocol by ID.

    Args:
        protocol_id: The protocol ID from protocols.io
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Dictionary containing detailed protocol information

    Raises:
        requests.RequestException: If API request fails
    """
    # Ensure access token is configured
    if not ACCESS_TOKEN:
        raise ValueError(
            "Protocols.io access token is not configured. Set PROTOCOLS_IO_ACCESS_TOKEN or BIOMNI_PROTOCOLS_IO_ACCESS_TOKEN env var, or configure BiomniConfig.protocols_io_access_token."
        )

    url = f"https://www.protocols.io/api/v3/protocols/{protocol_id}"

    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        data = response.json()

        if data.get("status_code") != 0:
            return {"error": data.get("error_message", "Unknown error"), "status_code": data.get("status_code")}

        return data.get("protocol", {})

    except requests.exceptions.RequestException as e:
        raise requests.RequestException(f"Failed to get protocol details: {str(e)}") from e
