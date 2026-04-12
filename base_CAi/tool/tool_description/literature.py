description = [
    {
        "description": "Fetches supplementary information for a paper given its DOI "
        "and saves it to a specified directory.",
        "name": "fetch_supplementary_info_from_doi",
        "optional_parameters": [
            {
                "default": "supplementary_info",
                "description": "Directory to save supplementary files",
                "name": "output_dir",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The paper DOI",
                "name": "doi",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query arXiv for papers based on the provided search query.",
        "name": "query_arxiv",
        "optional_parameters": [
            {
                "default": 10,
                "description": "The maximum number of papers to retrieve.",
                "name": "max_papers",
                "type": "int",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The search query string.",
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query Google Scholar for papers based on the provided search "
        "query and return the first search result.",
        "name": "query_scholar",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "The search query string.",
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query PubMed for papers based on the provided search query.",
        "name": "query_pubmed",
        "optional_parameters": [
            {
                "default": 10,
                "description": "The maximum number of papers to retrieve.",
                "name": "max_papers",
                "type": "int",
            },
            {
                "default": 3,
                "description": "Maximum number of retry attempts with modified queries.",
                "name": "max_retries",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The search query string.",
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Search using Google search and return formatted results.",
        "name": "search_google",
        "optional_parameters": [
            {
                "default": 3,
                "description": "Number of results to return",
                "name": "num_results",
                "type": "int",
            },
            {
                "default": "en",
                "description": "Language code for search results",
                "name": "language",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": 'The search query (e.g., "protocol text or search question")',
                "name": "query",
                "type": "str",
            }
        ],
    },
    {
        "description": "Extract the text content of a webpage using requests and BeautifulSoup.",
        "name": "extract_url_content",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Webpage URL to extract content from",
                "name": "url",
                "type": "str",
            }
        ],
    },
    {
        "description": "Extract text content from a PDF file.",
        "name": "extract_pdf_content",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "URL of the PDF file",
                "name": "url",
                "type": "str",
            }
        ],
    },
    {
        "description": "Initiate an advanced web search by launching a specialized agent to collect relevant information and citations through multiple rounds of web searches for a given query.",
        "name": "advanced_web_search_claude",
        "optional_parameters": [
            {
                "default": 1,
                "description": "Maximum number of searches",
                "name": "max_searches",
                "type": "int",
            },
            {
                "default": 3,
                "description": "Maximum number of retry attempts with modified queries.",
                "name": "max_retries",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The search query string.",
                "name": "query",
                "type": "str",
            }
        ],
    },
]
