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
    {
        "description": "List available protocol files in the local biomni/tool/protocols/ directory. Includes protocols from Addgene and Thermo Fisher Scientific.",
        "name": "list_local_protocols",
        "optional_parameters": [
            {
                "default": None,
                "description": "Filter by source directory (e.g., 'addgene' or 'thermofisher'). If None, lists all protocols.",
                "name": "source",
                "type": "str",
            }
        ],
        "required_parameters": [],
    },
    {
        "description": "Read the contents of a local protocol file from biomni/tool/protocols/. Use list_local_protocols() first to find available protocol filenames.",
        "name": "read_local_protocol",
        "optional_parameters": [
            {
                "default": None,
                "description": "Source directory (e.g., 'addgene' or 'thermofisher'). If None, searches all sources.",
                "name": "source",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Name of the protocol file (e.g., 'Addgene_ Protocol - How to Run an Agarose Gel.txt')",
                "name": "filename",
                "type": "str",
            }
        ],
    },
]
