description = [
    {
        "description": "Search protocols.io for public protocols matching a keyword.",
        "name": "search_protocols",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Most important keyword or phrase to search (title, description, authors)",
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Retrieve detailed metadata for a specific protocols.io protocol by ID.",
        "name": "get_protocol_details",
        "optional_parameters": [
            {
                "default": 30,
                "description": "Request timeout in seconds",
                "name": "timeout",
                "type": "int",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Numeric protocol ID from protocols.io",
                "name": "protocol_id",
                "type": "int",
            }
        ],
    },
]
