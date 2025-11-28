import os
import re
import time
from io import BytesIO
from urllib.parse import urljoin

import PyPDF2
import requests
from bs4 import BeautifulSoup

# from googlesearch import search


def fetch_supplementary_info_from_doi(doi: str, output_dir: str = "supplementary_info"):
    """
    Fetches supplementary information for a scientific paper using its DOI and downloads associated files.

    This function resolves a DOI to the publisher's webpage, searches for supplementary materials
    (such as data files, appendices, or additional documents), and downloads them to a local directory.
    It provides detailed logging of the entire process for research tracking purposes.

    Args:
        doi (str): The Digital Object Identifier (DOI) of the paper. Should be in the format
            "10.xxxx/xxxxx" (e.g., "10.1038/nature12373"). The function will attempt to resolve
            this DOI using the CrossRef API.
        output_dir (str, optional): The local directory path where supplementary files will be
            saved. Defaults to "supplementary_info". The directory will be created if it doesn't
            exist. Files are saved with their original names from the download URLs.

    Returns:
        str: A detailed research log as a newline-separated string containing:
            - DOI resolution status and publisher URL
            - List of discovered supplementary material links
            - Download status for each file (success/failure)
            - Summary of total files downloaded
            - Any error messages encountered during the process

        If no supplementary materials are found or if the DOI cannot be resolved,
        the log will contain appropriate error messages.

    Raises:
        requests.RequestException: If network requests fail (handled internally, logged to output)
        OSError: If the output directory cannot be created (handled internally, logged to output)

    Example:
        >>> log = fetch_supplementary_info_from_doi("10.1038/nature12373", "my_supplements")
        >>> print(log)
        Starting process for DOI: 10.1038/nature12373
        Resolved DOI to publisher page: https://www.nature.com/articles/nature12373
        Found supplementary material link: https://www.nature.com/articles/nature12373/supplementary-information
        Created output directory: my_supplements
        Downloaded file: my_supplements/supplementary-information.pdf
        Successfully downloaded 1 file(s).

    Note:
        - The function searches for links containing keywords: "supplementary", "supplemental", or "appendix"
        - Different publishers may structure their supplementary materials differently
        - Some supplementary materials may require authentication or subscription access
        - The function uses a generic User-Agent header to avoid basic bot detection
        - File names are derived from the URL path and may not always be descriptive
    """
    research_log = []
    research_log.append(f"Starting process for DOI: {doi}")

    # CrossRef API to resolve DOI to a publisher page
    crossref_url = f"https://doi.org/{doi}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(crossref_url, headers=headers)

    if response.status_code != 200:
        log_message = (
            f"Failed to resolve DOI: {doi}. Status Code: {response.status_code}"
        )
        research_log.append(log_message)
        return {"log": research_log, "files": []}

    publisher_url = response.url
    research_log.append(f"Resolved DOI to publisher page: {publisher_url}")

    # Fetch publisher page
    response = requests.get(publisher_url, headers=headers)
    if response.status_code != 200:
        log_message = f"Failed to access publisher page for DOI {doi}."
        research_log.append(log_message)
        return {"log": research_log, "files": []}

    # Parse page content
    soup = BeautifulSoup(response.content, "html.parser")
    supplementary_links = []

    # Look for supplementary materials by keywords or links
    for link in soup.find_all("a", href=True):
        href = link.get("href")
        text = link.get_text().lower()
        if "supplementary" in text or "supplemental" in text or "appendix" in text:
            full_url = urljoin(publisher_url, href)
            supplementary_links.append(full_url)
            research_log.append(f"Found supplementary material link: {full_url}")

    if not supplementary_links:
        log_message = f"No supplementary materials found for DOI {doi}."
        research_log.append(log_message)
        return research_log

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    research_log.append(f"Created output directory: {output_dir}")

    # Download supplementary materials
    downloaded_files = []
    for link in supplementary_links:
        file_name = os.path.join(output_dir, link.split("/")[-1])
        file_response = requests.get(link, headers=headers)
        if file_response.status_code == 200:
            with open(file_name, "wb") as f:
                f.write(file_response.content)
            downloaded_files.append(file_name)
            research_log.append(f"Downloaded file: {file_name}")
        else:
            research_log.append(f"Failed to download file from {link}")

    if downloaded_files:
        research_log.append(f"Successfully downloaded {len(downloaded_files)} file(s).")
    else:
        research_log.append(f"No files could be downloaded for DOI {doi}.")

    return "\n".join(research_log)


def query_arxiv(query: str, max_papers: int = 10) -> str:
    """
    Search arXiv repository for academic papers using a text-based query.

    This function queries the arXiv API to find relevant academic papers based on the provided
    search terms. Results are sorted by relevance and formatted for easy reading. The function
    handles various search formats including author names, titles, abstracts, and subject categories.

    Parameters
    ----------
    query : str
        The search query string. Can include:
        - Keywords from title or abstract (e.g., "machine learning neural networks")
        - Author names (e.g., "au:Smith" or "Smith, John")
        - Subject categories (e.g., "cat:cs.AI" for artificial intelligence)
        - Title searches (e.g., "ti:transformer")
        - Complex queries with boolean operators (e.g., "quantum AND computing")

    max_papers : int, optional
        Maximum number of papers to retrieve from the search results.
        Default is 10. arXiv API limits may apply for very large requests.
        Must be a positive integer.

    Returns
    -------
    str
        Formatted search results as a string containing:
        - Paper titles
        - Complete abstracts
        - Papers separated by double newlines for readability

        If no papers are found, returns "No papers found on arXiv."
        If an error occurs, returns an error message starting with "Error querying arXiv:"

    Raises
    ------
    Exception
        Various exceptions may be raised by the underlying arxiv library:
        - Network connectivity issues
        - Invalid query format
        - API rate limiting
        - Timeout errors
        All exceptions are caught and returned as formatted error messages.

    Examples
    --------
    >>> results = query_arxiv("transformer attention mechanism", max_papers=5)
    >>> print(results)
    Title: Attention Is All You Need
    Summary: The dominant sequence transduction models are based on complex...

    Title: BERT: Pre-training of Deep Bidirectional Transformers...
    Summary: We introduce a new language representation model called BERT...

    >>> results = query_arxiv("au:Hinton", max_papers=3)
    >>> print(results)
    Title: Deep Learning
    Summary: We review the recent progress in deep learning...

    >>> results = query_arxiv("invalid query with special characters @#$%", max_papers=1)
    >>> print(results)
    Error querying arXiv: Invalid query format...

    Notes
    -----
    - The function uses the arxiv Python library which interfaces with arXiv API v1
    - Results are sorted by relevance score as determined by arXiv's search algorithm
    - Search queries are case-insensitive
    - The arXiv API may have rate limits; excessive requests may result in temporary blocks
    - Some papers may have incomplete abstracts or metadata
    - The function requires an active internet connection
    """
    import arxiv

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query, max_results=max_papers, sort_by=arxiv.SortCriterion.Relevance
        )
        results = "\n\n".join(
            [
                f"Title: {paper.title}\nSummary: {paper.summary}"
                for paper in client.results(search)
            ]
        )
        return results if results else "No papers found on arXiv."
    except Exception as e:
        return f"Error querying arXiv: {e}"


def query_scholar(query: str) -> str:
    """
    Search Google Scholar for academic papers and return the first relevant result.

    This function uses the scholarly library to query Google Scholar's database of academic
    literature. It returns detailed information about the first (most relevant) paper found,
    including bibliographic data and abstract when available.

    Parameters
    ----------
    query : str
        The search query string. Can include:
        - Keywords from paper title or content (e.g., "deep learning computer vision")
        - Author names (e.g., "Geoffrey Hinton neural networks")
        - Specific paper titles (e.g., "Attention Is All You Need")
        - Institution names or journal names
        - Boolean search terms and phrases in quotes

        Note: Google Scholar's search algorithm interprets natural language queries
        and finds semantically related papers.

    Returns
    -------
    str
        Formatted information about the first search result containing:
        - Title: The full title of the paper
        - Year: Publication year
        - Venue: Journal, conference, or publication venue
        - Abstract: Paper abstract/summary (when available)

        If no results are found, returns "No results found on Google Scholar."
        If an error occurs, returns an error message starting with "Error querying Google Scholar:"

    Raises
    ------
    Exception
        Various exceptions may be caught and returned as error messages:
        - Network connectivity issues
        - Rate limiting from Google Scholar
        - Parsing errors from malformed responses
        - Timeout errors
        - Captcha challenges (Google Scholar may block automated requests)

    Examples
    --------
    >>> result = query_scholar("transformer neural machine translation")
    >>> print(result)
    Title: Attention Is All You Need
    Year: 2017
    Venue: Advances in Neural Information Processing Systems
    Abstract: The dominant sequence transduction models are based on complex...

    >>> result = query_scholar("Geoffrey Hinton backpropagation")
    >>> print(result)
    Title: Learning representations by back-propagating errors
    Year: 1986
    Venue: Nature
    Abstract: We describe a new learning procedure, back-propagation...

    >>> result = query_scholar("nonexistent paper title xyz123")
    >>> print(result)
    No results found on Google Scholar.

    Notes
    -----
    - Only returns the first (most relevant) result to avoid rate limiting
    - Google Scholar may implement anti-bot measures including CAPTCHAs
    - The scholarly library may need periodic updates to handle Google Scholar changes
    - Some papers may have incomplete metadata (missing abstracts, venues, etc.)
    - Results may include preprints, theses, books, and patents in addition to journal papers
    - Consider using delays between multiple calls to avoid being blocked
    - Academic institution IP addresses may have better access than residential IPs

    Warning
    -------
    Google Scholar has strict rate limiting and anti-automation policies. Excessive
    automated queries may result in temporary or permanent IP blocking. Use responsibly
    and consider implementing delays between requests.
    """
    from scholarly import ProxyGenerator, scholarly

    # Set up a ProxyGenerator object to use free proxies
    # This needs to be done only once per session
    pg = ProxyGenerator()
    pg.FreeProxies()
    scholarly.use_proxy(pg)
    try:
        search_query = scholarly.search_pubs(query)
        result = next(search_query, None)
        if result:
            return f"Title: {result['bib']['title']}\nYear: {result['bib']['pub_year']}\nVenue: {result['bib']['venue']}\nAbstract: {result['bib']['abstract']}"
        else:
            return "No results found on Google Scholar."
    except Exception as e:
        return f"Error querying Google Scholar: {e}"


def query_pubmed(query: str, max_papers: int = 10, max_retries: int = 3) -> str:
    """
    Search PubMed database for biomedical and life science literature with intelligent retry mechanism.

    This function queries the PubMed database (maintained by NCBI) for relevant scientific papers
    in biomedical and life sciences. It includes an intelligent retry system that progressively
    simplifies the query if no results are found, increasing the likelihood of finding relevant papers.

    Parameters
    ----------
    query : str
        The search query string. PubMed supports various search formats:
        - Medical Subject Headings (MeSH) terms (e.g., "diabetes mellitus"[MeSH])
        - Author names (e.g., "Smith J[Author]" or "Smith J")
        - Journal names (e.g., "Nature[Journal]")
        - Publication types (e.g., "randomized controlled trial"[Publication Type])
        - Field-specific searches (e.g., "cancer[Title]", "gene expression[Title/Abstract]")
        - Boolean operators (e.g., "diabetes AND insulin", "heart OR cardiac")
        - Date ranges (e.g., "2020:2023[Date - Publication]")
        - Free text searches combining multiple terms

    max_papers : int, optional
        Maximum number of papers to retrieve from search results. Default is 10.
        PubMed API can handle large requests, but consider rate limits for very high values.
        Must be a positive integer.

    max_retries : int, optional
        Maximum number of retry attempts if the initial query returns no results. Default is 3.
        The retry mechanism progressively simplifies the query by removing words from the end,
        making it more likely to find relevant papers for overly specific queries.

    Returns
    -------
    str
        Formatted search results containing for each paper:
        - Title: Complete paper title
        - Abstract: Full abstract text (when available)
        - Journal: Publication journal name
        - URL: Direct link to the PubMed entry (https://pubmed.ncbi.nlm.nih.gov/PMID/)

        Papers are separated by double newlines for readability.

        If no papers are found after all retry attempts, returns:
        "No papers found on PubMed after multiple query attempts."

        If an error occurs, returns an error message starting with "Error querying PubMed:"

    Raises
    ------
    Exception
        Various exceptions may be caught and returned as error messages:
        - Network connectivity issues
        - PubMed API rate limiting
        - Invalid query syntax
        - Timeout errors
        - Authentication issues (if email is invalid)

    Examples
    --------
    >>> results = query_pubmed("CRISPR gene editing", max_papers=5)
    >>> print(results)
    Title: CRISPR-Cas9 gene editing for sickle cell disease and β-thalassemia
    Abstract: CRISPR-Cas9 gene editing has emerged as a promising therapeutic approach...
    Journal: New England Journal of Medicine
    URL: https://pubmed.ncbi.nlm.nih.gov/31881138/

    Title: Genome editing with CRISPR-Cas nucleases...
    Abstract: The development of clustered regularly interspaced short palindromic...
    Journal: Nature
    URL: https://pubmed.ncbi.nlm.nih.gov/24505130/

    >>> results = query_pubmed("very specific rare disease xyz123", max_retries=2)
    >>> print(results)
    # First tries "very specific rare disease xyz123" -> no results
    # Then tries "very specific rare disease" -> no results
    # Then tries "very specific rare" -> finds results
    Title: Rare disease diagnosis using machine learning...
    Abstract: Rare diseases affect millions of people worldwide...

    >>> results = query_pubmed("Smith J[Author] AND diabetes", max_papers=3)
    >>> print(results)
    Title: Type 2 diabetes management in elderly patients
    Abstract: The management of type 2 diabetes in elderly patients...
    Journal: Diabetes Care
    URL: https://pubmed.ncbi.nlm.nih.gov/12345678/

    Notes
    -----
    - Uses the pymed library which interfaces with PubMed's E-utilities API
    - Includes 1-second delays between retry attempts to respect API rate limits
    - The retry mechanism removes the last word from the query with each attempt
    - Email address in the PubMed client should be updated to a valid address for production use
    - Some papers may have incomplete abstracts or missing metadata
    - PubMed primarily contains biomedical and life science literature
    - Results are not sorted by relevance; PubMed returns them in its default order
    - The function requires an active internet connection

    See Also
    --------
    query_arxiv : For physics, mathematics, and computer science papers
    query_scholar : For broader academic literature search across disciplines
    """
    from pymed import PubMed

    try:
        pubmed = PubMed(
            tool="MyTool", email="your-email@example.com"
        )  # Update with a valid email address

        # Initial attempt
        papers = list(pubmed.query(query, max_results=max_papers))

        # Retry with modified queries if no results
        retries = 0
        while not papers and retries < max_retries:
            retries += 1
            # Simplify query with each retry by removing the last word
            simplified_query = (
                " ".join(query.split()[:-retries])
                if len(query.split()) > retries
                else query
            )
            time.sleep(1)  # Add delay between requests
            papers = list(pubmed.query(simplified_query, max_results=max_papers))

        if papers:
            results = []
            for paper in papers:
                pubmed_id = paper.pubmed_id.split("\n")[0]
                content = f"Title: {paper.title}\n"
                content += f"Abstract: {paper.abstract}\n"
                content += f"Journal: {paper.journal}\n"
                content += f"URL: https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/\n"
                results.append(content)
            results = "\n\n".join(results)
            return results
        else:
            return "No papers found on PubMed after multiple query attempts."
    except Exception as e:
        return f"Error querying PubMed: {e}"


def search_google(query: str, num_results: int = 3, language: str = "en") -> str:
    """
    Perform a Google web search and return formatted results with titles, URLs, and descriptions.

    This function uses the googlesearch-python library to query Google's search engine
    and retrieve web results. It's particularly useful for finding protocols, research
    information, and general web content related to scientific queries.

    Args:
        query (str): The search query string. Can include:
            - Natural language questions (e.g., "How to perform PCR amplification?")
            - Protocol names (e.g., "Western blot protocol")
            - Scientific terms and concepts (e.g., "CRISPR genome editing methods")
            - Specific product or technique names
            - Site-specific searches using "site:" operator (e.g., "site:nature.com gene editing")
            - Quoted phrases for exact matches (e.g., '"protein purification protocol"')

        num_results (int, optional): Number of search results to return. Default is 3.
            Higher values may increase the risk of rate limiting. Recommended range: 1-10.

        language (str, optional): Language code for search results. Default is "en" (English).
            Common codes include:
            - "en": English
            - "es": Spanish
            - "fr": French
            - "de": German
            - "ja": Japanese
            - "zh": Chinese

    Returns:
        str: Formatted search results as a string containing for each result:
            - Title: The webpage title
            - URL: The full URL of the webpage
            - Description: Brief description or snippet from the page

            Results are separated by double newlines for readability.

            If an error occurs during search, returns an error message.

    Raises:
        Exception: Various exceptions may occur and are handled internally:
            - Network connectivity issues
            - Rate limiting from Google (HTTP 429 errors)
            - Invalid query format
            - Blocked requests (HTTP 403 errors)
            - Timeout errors

    Examples:
        >>> results = search_google("PCR protocol molecular biology", num_results=2)
        >>> print(results)
        Title: PCR Protocol - Thermo Fisher Scientific
        URL: https://www.thermofisher.com/pcr-protocol
        Description: Step-by-step PCR protocol for molecular biology applications...

        Title: Polymerase Chain Reaction (PCR) - Protocol Online
        URL: https://www.protocol-online.org/biology-forums-lab-techniques/posts/7234.html
        Description: Detailed PCR protocol with troubleshooting tips and optimization...

        >>> results = search_google("site:ncbi.nlm.nih.gov gene expression", num_results=1, language="en")
        >>> print(results)
        Title: Gene Expression - NCBI
        URL: https://www.ncbi.nlm.nih.gov/gene/expression
        Description: Tools and databases for gene expression analysis...

        >>> results = search_google("Western blot protocol", num_results=3, language="en")
        >>> print(results)
        Title: Western Blot Protocol - Bio-Rad
        URL: https://www.bio-rad.com/western-blot-protocol
        Description: Complete western blot protocol from sample preparation to detection...

    Notes:
        - Uses the googlesearch-python library with advanced search features enabled
        - Includes automatic rate limiting protection to avoid being blocked
        - Search results quality depends on Google's algorithm and current indexing
        - Some results may be behind paywalls or require authentication
        - The function includes debug print statements for monitoring search progress
        - Consider using delays between multiple consecutive searches
        - Google may block automated searches from certain IP addresses

    Warning:
        Google has terms of service that restrict automated searching. Use this function
        responsibly and in compliance with Google's robots.txt and terms of service.
        Excessive automated queries may result in temporary or permanent IP blocking.

    See Also:
        advanced_web_search_claude : For more sophisticated web searches with AI assistance
        extract_url_content : For extracting content from the returned URLs
    """
    return search_duckduckgo(query, num_results, language)
    from googlesearch import search

    try:
        results_string = ""
        search_query = f"{query}"

        print(
            f"Searching for {search_query} with {num_results} results and {language} language"
        )

        for res in search(search_query, num=num_results, lang=language):
            print(f"Found result: {res.title}")
            title = res.title
            url = res.url
            description = res.description

            results_string += (
                f"Title: {title}\nURL: {url}\nDescription: {description}\n\n"
            )

        return results_string

    except Exception as e:
        return f"Error performing search: {str(e)}"


def search_duckduckgo(query: str, num_results: int = 3, language: str = "en") -> str:
    """
    Perform a DuckDuckGo web search and return formatted results with titles, URLs, and descriptions.

    This function uses the duckduckgo-search library to query DuckDuckGo's search engine
    and retrieve web results. It's more reliable than Google for automated queries as it
    doesn't have rate limiting or blocking issues. Particularly useful for finding protocols,
    research information, and general web content related to scientific queries.

    Args:
        query (str): The search query string. Can include:
            - Natural language questions (e.g., "How to perform PCR amplification?")
            - Protocol names (e.g., "Western blot protocol")
            - Scientific terms and concepts (e.g., "CRISPR genome editing methods")
            - Specific product or technique names
            - Site-specific searches using "site:" operator (e.g., "site:nature.com gene editing")
            - Quoted phrases for exact matches (e.g., '"protein purification protocol"')

        num_results (int, optional): Number of search results to return. Default is 3.
            Recommended range: 1-20.

        language (str, optional): Language code for search results. Default is "en" (English).
            Common codes include:
            - "en": English
            - "es": Spanish
            - "fr": French
            - "de": German
            - "ja": Japanese
            - "zh": Chinese

    Returns:
        str: Formatted search results as a string containing for each result:
            - Title: The webpage title
            - URL: The full URL of the webpage
            - Description: Brief description or snippet from the page

            Results are separated by double newlines for readability.

            If no results are found, returns "No results found."
            If an error occurs during search, returns an error message.

    Raises:
        Exception: Various exceptions may occur and are handled internally:
            - Network connectivity issues
            - Invalid query format
            - Timeout errors
            - ImportError if duckduckgo-search library is not installed

    Examples:
        >>> results = search_duckduckgo("PCR protocol molecular biology", num_results=2)
        >>> print(results)
        Result 1:
        Title: PCR Protocol - Thermo Fisher Scientific
        URL: https://www.thermofisher.com/pcr-protocol
        Description: Step-by-step PCR protocol for molecular biology applications...

        Result 2:
        Title: Polymerase Chain Reaction (PCR) - Protocol Online
        URL: https://www.protocol-online.org/biology-forums-lab-techniques/posts/7234.html
        Description: Detailed PCR protocol with troubleshooting tips and optimization...

        >>> results = search_duckduckgo("site:ncbi.nlm.nih.gov gene expression", num_results=1)
        >>> print(results)
        Result 1:
        Title: Gene Expression - NCBI
        URL: https://www.ncbi.nlm.nih.gov/gene/expression
        Description: Tools and databases for gene expression analysis...

        >>> results = search_duckduckgo("Western blot protocol", num_results=3, language="en")
        >>> print(results)
        Result 1:
        Title: Western Blot Protocol - Bio-Rad
        URL: https://www.bio-rad.com/western-blot-protocol
        Description: Complete western blot protocol from sample preparation to detection...

    Notes:
        - Uses DuckDuckGo search via duckduckgo-search library for better reliability
        - No rate limiting or IP blocking issues compared to Google automated searches
        - Search results quality is generally good and comparable to Google
        - Some results may be behind paywalls or require authentication
        - The function includes debug print statements for monitoring search progress
        - Works reliably in automated environments without requiring API keys
        - Privacy-friendly as DuckDuckGo doesn't track searches

    Installation:
        If the library is not installed, you can install it with:
        pip install duckduckgo-search

    See Also:
        search_google : For Google web searches (may have blocking issues)
        advanced_web_search_claude : For more sophisticated web searches with AI assistance
        extract_url_content : For extracting content from the returned URLs
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: duckduckgo_search library not installed. Install with: pip install ddgs"

    try:
        results_string = ""

        print(
            f"Searching for '{query}' with {num_results} results and {language} language"
        )

        with DDGS() as ddgs:
            # Use region parameter for language (e.g., "wt-en" for worldwide English)
            region = f"wt-{language}" if language else "wt-en"
            search_results = list(
                ddgs.text(query, max_results=num_results, region=region)
            )

            if not search_results:
                return "No results found."

            for idx, result in enumerate(search_results):
                title = result.get("title", "N/A")
                url = result.get("href", "N/A")
                description = result.get("body", "N/A")

                print(f"Found result {idx+1}: {title}")

                results_string += f"Result {idx+1}:\n"
                results_string += f"Title: {title}\n"
                results_string += f"URL: {url}\n"
                results_string += f"Description: {description}\n\n"

        return results_string

    except Exception as e:
        return f"Error performing DuckDuckGo search: {str(e)}"


def advanced_web_search_claude(
    query: str,
    max_searches: int = 1,
    max_retries: int = 3,
) -> str:
    """
    Perform advanced web search using Claude AI with integrated web search capabilities.

    This function leverages Claude's web search tool to conduct intelligent, multi-step
    web searches that go beyond simple keyword matching. Claude can understand context,
    follow up on initial results, and synthesize information from multiple sources to
    provide comprehensive answers with proper citations.

    Parameters
    ----------
    query : str
        The search query or question for Claude to investigate. Should be:
        - Specific and well-formulated (e.g., "What are the latest developments in mRNA vaccine technology?")
        - Complex questions that benefit from AI analysis (e.g., "Compare different CRISPR delivery methods for in vivo applications")
        - Research questions requiring synthesis of multiple sources
        - Topics needing current, up-to-date information
        - Scientific protocols or methodology questions

        Avoid overly broad queries; be specific about what information you're seeking.

    max_searches : int, optional
        Maximum number of web searches Claude can perform for this query. Default is 1.
        Higher values allow for more comprehensive research but increase cost and time:
        - 1-2: Quick, focused searches for straightforward questions
        - 3-5: Moderate research requiring multiple sources
        - 5+: Comprehensive research on complex topics

    max_retries : int, optional
        Maximum number of retry attempts if the initial request fails. Default is 3.
        Uses exponential backoff (1s, 2s, 4s delays) between retries to handle:
        - Temporary network issues
        - API rate limiting
        - Transient service errors

    Returns
    -------
    str
        A comprehensive formatted response containing:
        - Claude's synthesized analysis of the search results
        - Key findings and insights
        - Inline citations with source titles and URLs in the format:
          "(Citation: Source Title - URL)"
        - Structured information organized by topic or relevance

        If an error occurs after all retry attempts, returns an error message
        starting with "Error performing web search after X attempts:"

    Raises
    ------
    ValueError
        - If the model is not a Claude model (must contain "claude" in the name)
        - If no API key is provided or found in environment variables

    Exception
        Various exceptions may occur during web search:
        - Network connectivity issues
        - Anthropic API rate limiting
        - Invalid API key or authentication errors
        - Timeout errors
        - Service unavailability

    Configuration Requirements
    -------------------------
    This function requires either:
    1. A configured biomni.config.default_config with:
       - llm: Claude model name (e.g., "claude-4-sonnet-latest")
       - api_key: Valid Anthropic API key
    2. Or environment variable ANTHROPIC_API_KEY set to a valid API key

    Examples
    --------
    >>> response = advanced_web_search_claude("latest CRISPR base editing techniques 2024")
    >>> print(response)
    Based on recent research, several advanced CRISPR base editing techniques have emerged in 2024...

    Prime editing has been enhanced with improved efficiency through... (Citation: Nature Biotechnology - https://www.nature.com/articles/...)

    New cytosine base editors with reduced off-target effects... (Citation: Science - https://www.science.org/doi/...)

    >>> response = advanced_web_search_claude(
    ...     "How do different COVID-19 vaccine platforms compare in terms of efficacy and safety?",
    ...     max_searches=3
    ... )
    >>> print(response)
    COVID-19 vaccines can be categorized into several platforms, each with distinct characteristics:

    mRNA vaccines (Pfizer-BioNTech, Moderna) show... (Citation: NEJM - https://www.nejm.org/doi/...)
    Viral vector vaccines (Johnson & Johnson, AstraZeneca)... (Citation: The Lancet - https://www.thelancet.com/journals/...)

    >>> response = advanced_web_search_claude("nonexistent topic xyz123 research")
    >>> print(response)
    I searched for information about "nonexistent topic xyz123 research" but was unable to find any relevant...

    Notes
    -----
    - Uses Anthropic's Claude model with the web_search_20250305 tool
    - Includes random delays (1-10 seconds) before requests to avoid overwhelming the API
    - Citations are automatically extracted and formatted inline with the response
    - The function is designed for research and information gathering, not real-time data
    - Results quality depends on Claude's ability to understand the query and find relevant sources
    - More complex queries generally benefit from higher max_searches values
    - Consider the cost implications of higher max_searches values

    Performance Tips
    ----------------
    - Craft specific, well-structured queries for best results
    - Use higher max_searches (3-5) for complex, multi-faceted questions
    - Keep max_searches low (1-2) for straightforward factual queries
    - Include relevant keywords and context in your query
    - Specify time frames if current information is needed (e.g., "2024 developments in...")

    See Also
    --------
    search_google : For basic Google web searches without AI analysis
    query_pubmed : For biomedical literature searches
    query_arxiv : For academic paper searches in physics/math/CS
    """
    import random

    import anthropic

    try:
        from biomni.config import default_config

        model = default_config.llm
        api_key = default_config.api_key
        if not api_key:
            api_key = os.getenv("ANTHROPIC_API_KEY")
    except ImportError:
        model = "claude-4-sonnet-latest"
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if "claude" not in model:
        raise ValueError("Model must be a Claude model.")

    if not api_key:
        raise ValueError("Set your api_key explicitly.")

    client = anthropic.Anthropic(api_key=api_key)
    tool_def = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_searches,
    }

    delay = random.randint(1, 10)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": query}],
                tools=[tool_def],
            )

            paragraphs, citations = [], []
            response.content = response.content
            formatted_response = ""
            for blk in response.content:
                if blk.type == "text":
                    paragraphs.append(blk.text)
                    formatted_response += blk.text

                    if blk.citations:
                        for cite in blk.citations:
                            citations.append(
                                {
                                    "url": cite.url,
                                    "title": cite.title,
                                    "cited_text": cite.cited_text,
                                }
                            )
                            formatted_response += (
                                f"(Citation: {cite.title} - {cite.url})"
                            )
            return formatted_response

        except Exception as e:
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            print(f"Error performing web search after {max_retries} attempts: {str(e)}")
            return f"Error performing web search after {max_retries} attempts: {str(e)}"


def extract_url_content(url: str) -> str:
    """
    Extract clean, readable text content from a webpage URL using intelligent parsing.

    This function fetches webpage content and intelligently extracts the main text
    while filtering out navigation, advertisements, and other non-content elements.
    It handles different content types and provides structured text output suitable
    for analysis or reading.

    Args:
        url (str): The complete URL of the webpage to extract content from.
            Should include the protocol (http:// or https://). Examples:
            - "https://www.nature.com/articles/nature12373"
            - "https://en.wikipedia.org/wiki/CRISPR"
            - "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/"
            - "https://protocols.io/view/pcr-protocol-xyz123"

    Returns:
        str: Clean, formatted text content of the webpage containing:
            - Main article or content text
            - Headings and subheadings preserved
            - Paragraphs separated by double newlines for readability
            - Navigation, ads, scripts, and styling removed

            For plain text or JSON responses, returns the raw content directly.

            If the page cannot be accessed or parsed, may return an empty string
            or minimal content depending on the webpage structure.

    Raises:
        requests.RequestException: If the URL cannot be accessed due to:
            - Network connectivity issues
            - Invalid URL format
            - HTTP errors (404, 403, 500, etc.)
            - Timeout errors

        Exception: Other parsing errors that may occur during content extraction

    Content Type Handling:
        - **HTML pages**: Parsed with BeautifulSoup, main content extracted
        - **Plain text**: Returned directly without modification
        - **JSON**: Returned as formatted JSON string
        - **Other formats**: Attempted HTML parsing as fallback

    Parsing Strategy:
        1. Attempts to find main content areas: <main>, <article>, or <body>
        2. Removes unwanted elements: scripts, styles, navigation, headers, footers, sidebars, iframes
        3. Extracts text from paragraphs and headings (p, h1-h6 tags)
        4. Formats output with proper spacing and line breaks

    Examples:
        >>> content = extract_url_content("https://en.wikipedia.org/wiki/PCR")
        >>> print(content[:200])
        Polymerase Chain Reaction

        The polymerase chain reaction (PCR) is a method widely used to rapidly make millions...

        History

        The PCR technique was developed in 1983 by American biochemist Kary Mullis...

        >>> content = extract_url_content("https://www.nature.com/articles/nature12373")
        >>> print(content[:150])
        CRISPR-Cas9 genome editing in human cells

        The CRISPR-Cas9 system has emerged as a powerful tool for genome editing...

        >>> content = extract_url_content("https://api.example.com/data.json")
        >>> print(content)
        {"status": "success", "data": {"experiments": [...], "results": [...]}}

        >>> content = extract_url_content("https://example.com/plain-text-protocol.txt")
        >>> print(content)
        PCR Protocol
        Materials needed:
        - DNA template
        - Primers (forward and reverse)
        - DNA polymerase...

    Notes:
        - Uses a generic Mozilla User-Agent to avoid basic bot detection
        - Focuses on extracting meaningful content while preserving structure
        - May not work perfectly with heavily JavaScript-dependent pages
        - Some content may be behind authentication or paywalls
        - The quality of extraction depends on the webpage's HTML structure
        - Large pages may take longer to process due to parsing overhead

    Limitations:
        - Cannot handle JavaScript-rendered content (SPA applications)
        - May miss content that loads dynamically after page load
        - Some websites may block automated requests
        - Complex layouts may result in suboptimal text extraction
        - Does not handle multimedia content (images, videos, etc.)

    See Also:
        extract_pdf_content : For extracting text from PDF files
        search_google : For finding relevant webpages
        advanced_web_search_claude : For AI-powered web content analysis
    """
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

    # Check if the response is in text format
    if "text/plain" in response.headers.get(
        "Content-Type", ""
    ) or "application/json" in response.headers.get("Content-Type", ""):
        return response.text.strip()  # Return plain text or JSON response directly

    # If it's HTML, use BeautifulSoup to parse
    soup = BeautifulSoup(response.text, "html.parser")

    # Try to find main content first, fallback to body
    content = soup.find("main") or soup.find("article") or soup.body

    # Remove unwanted elements
    for element in content(
        ["script", "style", "nav", "header", "footer", "aside", "iframe"]
    ):
        element.decompose()

    # Extract text with better formatting
    paragraphs = content.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    cleaned_text = []

    for p in paragraphs:
        text = p.get_text().strip()
        if text:  # Only add non-empty paragraphs
            cleaned_text.append(text)

    return "\n\n".join(cleaned_text)


def extract_pdf_content(url: str) -> str:
    """
    Extract text content from a PDF file accessible via URL with intelligent PDF detection.

    This function downloads and extracts readable text from PDF files, handling both
    direct PDF URLs and web pages containing PDF links. It uses PyPDF2 for text
    extraction and includes validation to ensure the downloaded content is actually
    a PDF file.

    Args:
        url (str): URL pointing to a PDF file or webpage containing PDF links.
            Supported formats:
            - Direct PDF URLs: "https://example.com/paper.pdf"
            - Journal article pages with PDF links: "https://www.nature.com/articles/nature12373"
            - Repository URLs: "https://arxiv.org/pdf/2103.00020.pdf"
            - Protocol sites: "https://protocols.io/view/protocol.pdf"

            The function will automatically detect and follow PDF links if the
            provided URL is not a direct PDF link.

    Returns:
        str: Extracted text content from the PDF containing:
            - All readable text from every page of the PDF
            - Pages separated by double newlines
            - Whitespace normalized (multiple spaces/newlines reduced)
            - Special characters and formatting preserved where possible

            If extraction fails or no text is found, returns descriptive error messages:
            - "No PDF file found at {url}. Please provide a direct link to a PDF file."
            - "The URL did not return a valid PDF file. Content type: {content_type}"
            - "The PDF file did not contain any extractable text. It may be an image-based PDF requiring OCR."
            - "Error downloading PDF: {error_message}"
            - "Error extracting text from PDF: {error_message}"

    Raises:
        requests.exceptions.RequestException: Network-related errors including:
            - Connection timeouts (30 second timeout)
            - HTTP errors (404, 403, 500, etc.)
            - Network connectivity issues
            - Invalid URLs

        Exception: PDF processing errors including:
            - Corrupted PDF files
            - Password-protected PDFs
            - Unsupported PDF versions
            - Memory issues with very large PDFs

    PDF Detection and Validation:
        1. **Direct PDF URLs**: If URL ends with .pdf, downloads directly
        2. **Indirect URLs**: Searches HTML content for PDF links using regex
        3. **Content validation**: Checks both Content-Type header and PDF magic bytes (%PDF)
        4. **Relative URL handling**: Converts relative PDF paths to absolute URLs

    Text Extraction Process:
        1. Downloads PDF content into memory (BytesIO buffer)
        2. Validates PDF format using magic bytes and content type
        3. Uses PyPDF2.PdfReader to parse PDF structure
        4. Extracts text from each page sequentially
        5. Normalizes whitespace and formatting
        6. Combines all pages with double newline separators

    Examples:
        >>> content = extract_pdf_content("https://arxiv.org/pdf/1706.03762.pdf")
        >>> print(content[:200])
        Attention Is All You Need

        Abstract
        The dominant sequence transduction models are based on complex recurrent or
        convolutional neural networks that include an encoder and decoder...

        >>> content = extract_pdf_content("https://www.nature.com/articles/nature12373")
        >>> print(content[:150])
        # Function finds PDF link on the Nature article page and extracts:
        CRISPR-Cas9 genome editing in human cells

        We describe a set of tools for genome editing based on the Cas9 nuclease...

        >>> content = extract_pdf_content("https://example.com/not-a-pdf")
        >>> print(content)
        No PDF file found at https://example.com/not-a-pdf. Please provide a direct link to a PDF file.

        >>> content = extract_pdf_content("https://example.com/scanned-document.pdf")
        >>> print(content)
        The PDF file did not contain any extractable text. It may be an image-based PDF requiring OCR.

    Notes:
        - Uses 30-second timeout for downloads to handle large PDF files
        - Automatically handles relative URLs by constructing absolute paths
        - Searches for PDF links using case-insensitive regex pattern
        - Memory-efficient processing using BytesIO for large files
        - Whitespace normalization improves readability of extracted text
        - Works best with text-based PDFs; scanned documents may not extract properly

    Limitations:
        - Cannot extract text from image-based or scanned PDFs (requires OCR)
        - Password-protected PDFs will fail extraction
        - Very large PDFs may cause memory issues
        - Complex formatting (tables, columns) may not preserve structure
        - Mathematical equations and special symbols may not extract correctly
        - Some PDF encryption or protection schemes may prevent text extraction

    Performance Considerations:
        - Large PDFs (>100MB) may take significant time to download and process
        - Network speed affects download time for remote PDFs
        - Complex PDFs with many pages may have slower text extraction
        - Consider implementing caching for frequently accessed PDFs

    See Also:
        extract_url_content : For extracting content from web pages
        fetch_supplementary_info_from_doi : For downloading supplementary PDF materials
        query_arxiv : For finding and accessing arXiv PDF papers
    """
    try:
        # Check if the URL ends with .pdf
        if not url.lower().endswith(".pdf"):
            # If not, try to find a PDF link on the page
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Look for PDF links in the HTML content
                pdf_links = re.findall(r'href=[\'"]([^\'"]+\.pdf)[\'"]', response.text)
                if pdf_links:
                    # Use the first PDF link found
                    if not pdf_links[0].startswith("http"):
                        # Handle relative URLs
                        base_url = "/".join(url.split("/")[:3])
                        url = (
                            base_url + pdf_links[0]
                            if pdf_links[0].startswith("/")
                            else base_url + "/" + pdf_links[0]
                        )
                    else:
                        url = pdf_links[0]
                else:
                    return f"No PDF file found at {url}. Please provide a direct link to a PDF file."

        # Download the PDF
        response = requests.get(url, timeout=30)

        # Check if we actually got a PDF file (by checking content type or magic bytes)
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type and not response.content.startswith(
            b"%PDF"
        ):
            return (
                f"The URL did not return a valid PDF file. Content type: {content_type}"
            )

        pdf_file = BytesIO(response.content)

        # Try with PyPDF2 first
        try:
            text = ""
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")

        # Clean up the text
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return "The PDF file did not contain any extractable text. It may be an image-based PDF requiring OCR."

        return text

    except requests.exceptions.RequestException as e:
        return f"Error downloading PDF: {str(e)}"
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"
