import os
import re
import time
from io import BytesIO
from urllib.parse import urljoin

import PyPDF2
import requests
from bs4 import BeautifulSoup
from googlesearch import search


def fetch_supplementary_info_from_doi(doi: str, output_dir: str = "supplementary_info"):
    """Fetches supplementary information for a paper given its DOI and returns a research log.

    Args:
        doi: The paper DOI.
        output_dir: Directory to save supplementary files.

    Returns:
        dict: A dictionary containing a research log and the downloaded file paths.

    """
    research_log = []
    research_log.append(f"Starting process for DOI: {doi}")

    # CrossRef API to resolve DOI to a publisher page
    crossref_url = f"https://doi.org/{doi}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(crossref_url, headers=headers)

    if response.status_code != 200:
        log_message = f"Failed to resolve DOI: {doi}. Status Code: {response.status_code}"
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
    """Query arXiv for papers based on the provided search query.

    Parameters
    ----------
    - query (str): The search query string.
    - max_papers (int): The maximum number of papers to retrieve (default: 10).

    Returns
    -------
    - str: The formatted search results or an error message.

    """
    import arxiv

    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_papers, sort_by=arxiv.SortCriterion.Relevance)
        results = "\n\n".join([f"Title: {paper.title}\nSummary: {paper.summary}" for paper in client.results(search)])
        return results if results else "No papers found on arXiv."
    except Exception as e:
        return f"Error querying arXiv: {e}"


def query_scholar(query: str) -> str:
    """Query Google Scholar for papers based on the provided search query.

    Parameters
    ----------
    - query (str): The search query string.

    Returns
    -------
    - str: The first search result formatted or an error message.

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
    """Query PubMed for papers based on the provided search query.

    Parameters
    ----------
    - query (str): The search query string.
    - max_papers (int): The maximum number of papers to retrieve (default: 10).
    - max_retries (int): Maximum number of retry attempts with modified queries (default: 3).

    Returns
    -------
    - str: The formatted search results or an error message.

    """
    from pymed import PubMed

    try:
        pubmed = PubMed(tool="MyTool", email="your-email@example.com")  # Update with a valid email address

        # Initial attempt
        papers = list(pubmed.query(query, max_results=max_papers))

        # Retry with modified queries if no results
        retries = 0
        while not papers and retries < max_retries:
            retries += 1
            # Simplify query with each retry by removing the last word
            simplified_query = " ".join(query.split()[:-retries]) if len(query.split()) > retries else query
            time.sleep(1)  # Add delay between requests
            papers = list(pubmed.query(simplified_query, max_results=max_papers))

        if papers:
            results = "\n\n".join(
                [f"Title: {paper.title}\nAbstract: {paper.abstract}\nJournal: {paper.journal}" for paper in papers]
            )
            return results
        else:
            return "No papers found on PubMed after multiple query attempts."
    except Exception as e:
        return f"Error querying PubMed: {e}"


def search_google(query: str, num_results: int = 3, language: str = "en") -> list[dict]:
    """Search using Google search.

    Args:
        query (str): The search query (e.g., "protocol text or seach question")
        num_results (int): Number of results to return (default: 10)
        language (str): Language code for search results (default: 'en')
        pause (float): Pause between searches to avoid rate limiting (default: 2.0 seconds)

    Returns:
        List[dict]: List of dictionaries containing search results with title and URL

    """
    try:
        results_string = ""
        search_query = f"{query}"

        print(f"Searching for {search_query} with {num_results} results and {language} language")

        for res in search(search_query, num_results=num_results, lang=language, advanced=True):
            print(f"Found result: {res.title}")
            title = res.title
            url = res.url
            description = res.description

            results_string += f"Title: {title}\nURL: {url}\nDescription: {description}\n\n"

    except Exception as e:
        print(f"Error performing search: {str(e)}")
    return results_string


def advanced_web_search_claude(
    query: str,
    max_searches: int = 1,
    max_retries: int = 3,
) -> tuple[str, list[dict[str, str]], list]:
    """
    Initiate an advanced web search by launching a specialized agent to collect relevant information and citations through multiple rounds of web searches for a given query.
    Craft the query carefully for the search agent to find the most relevant information.

    Parameters
    ----------
    query : str
        The search phrase you want Claude to look up.
    max_searches : int, optional
        Upper-bound on searches Claude may issue inside this request.
    max_retries : int, optional
        Maximum number of retry attempts with exponential backoff.

    Returns
    -------
    full_text : str
        A formatted string containing the full text response from Claude and the citations.
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
                            citations.append({"url": cite.url, "title": cite.title, "cited_text": cite.cited_text})
                            formatted_response += f"(Citation: {cite.title} - {cite.url})"
            return formatted_response

        except Exception as e:
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            print(f"Error performing web search after {max_retries} attempts: {str(e)}")
            return f"Error performing web search after {max_retries} attempts: {str(e)}"


def extract_url_content(url: str) -> str:
    """Extract the text content of a webpage using requests and BeautifulSoup.

    Args:
        url: Webpage URL to extract content from

    Returns:
        Text content of the webpage

    """
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

    # Check if the response is in text format
    if "text/plain" in response.headers.get("Content-Type", "") or "application/json" in response.headers.get(
        "Content-Type", ""
    ):
        return response.text.strip()  # Return plain text or JSON response directly

    # If it's HTML, use BeautifulSoup to parse
    soup = BeautifulSoup(response.text, "html.parser")

    # Try to find main content first, fallback to body
    content = soup.find("main") or soup.find("article") or soup.body

    # Remove unwanted elements
    for element in content(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
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
    """Extract the text content of a PDF file given its URL.

    Args:
        url: URL of the PDF file to extract text from

    Returns:
        The extracted text content from the PDF

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
                        url = base_url + pdf_links[0] if pdf_links[0].startswith("/") else base_url + "/" + pdf_links[0]
                    else:
                        url = pdf_links[0]
                else:
                    return f"No PDF file found at {url}. Please provide a direct link to a PDF file."

        # Download the PDF
        response = requests.get(url, timeout=30)

        # Check if we actually got a PDF file (by checking content type or magic bytes)
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type and not response.content.startswith(b"%PDF"):
            return f"The URL did not return a valid PDF file. Content type: {content_type}"

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


_FIRECRAWL_API_URL = "https://api.firecrawl.dev/v2"


def _firecrawl_request(endpoint: str, payload: dict, timeout: int = 120) -> dict | str:
    """POST to the Firecrawl API and return the parsed body, or an error string.

    Works without an API key on Firecrawl's keyless tier, which is capped per IP address per day. If
    FIRECRAWL_API_KEY is set it is sent as a Bearer token and requests count against that account's plan
    limits instead. Firecrawl reports some failures in the body as {"success": false, "error": ...} with
    HTTP 200 (DNS errors on scrape, for example), so the body is read before the status code.
    """
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.post(f"{_FIRECRAWL_API_URL}/{endpoint}", headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as e:
        return f"Error reaching the Firecrawl API: {str(e)}"

    try:
        body = response.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}

    if response.status_code == 429 and not api_key:
        reason = body.get("reason") or "rate limit"
        retry = body.get("retry_after_seconds")
        wait = f"Retry in about {retry} seconds" if isinstance(retry, int) else "Retry later"
        return (
            f"Firecrawl keyless limit reached for this IP address ({reason}). {wait}, "
            "or set FIRECRAWL_API_KEY for higher limits."
        )
    if response.status_code == 401:
        return "Firecrawl rejected FIRECRAWL_API_KEY (401 Unauthorized). Check the key or unset it to use the keyless tier."
    if not body.get("success", False):
        error = str(body.get("error") or response.text[:300] or "no error message")
        if api_key:
            error = error.replace(api_key, "[REDACTED]")
        status = "" if response.status_code == 200 else f" (HTTP {response.status_code})"
        return f"Firecrawl API error{status}: {error}"
    return body


def firecrawl_search(query: str, num_results: int = 5, scrape_content: bool = False) -> str:
    """Search the web with the Firecrawl Search API and return formatted results.

    Uses the Firecrawl Search API (https://docs.firecrawl.dev/features/search). With scrape_content=True,
    each result also includes the page content as markdown. JavaScript-rendered pages and PDFs are handled.

    Works without an API key (keyless tier, capped per IP per day). Set FIRECRAWL_API_KEY for higher limits.

    Args:
        query (str): The search query (e.g., "protocol text or search question")
        num_results (int): Number of results to return (default: 5, max: 20)
        scrape_content (bool): Also fetch each result page as markdown (default: False)

    Returns:
        str: Results formatted as "Title / URL / Description" blocks, plus "Content" when
        scrape_content is True, or an error message

    """
    if not isinstance(query, str) or not query.strip():
        return "Error: query must be a non-empty string."
    try:
        limit = max(1, min(int(num_results), 20))
    except (TypeError, ValueError):
        return f"Error: num_results must be an integer, got {num_results!r}."

    per_result_chars = 4000  # keeps each Content block readable for the agent
    total_content_chars = 20000  # and bounds the whole observation

    payload = {
        "query": query,
        "limit": limit,
        "sources": ["web"],
        "highlights": False,  # plain snippets, same shape as search_google
    }
    if scrape_content:
        payload["scrapeOptions"] = {"formats": ["markdown"], "onlyMainContent": True}

    body = _firecrawl_request("search", payload)
    if isinstance(body, str):
        return body

    data = body.get("data")
    results = data.get("web") if isinstance(data, dict) else None
    if not isinstance(results, list) or not results:
        return "No results found on Firecrawl search."

    results_string = ""
    for res in results:
        if not isinstance(res, dict):
            continue
        metadata = res.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        title = res.get("title") or metadata.get("title") or ""
        url = res.get("url") or metadata.get("sourceURL") or ""
        description = res.get("description") or metadata.get("description") or ""

        results_string += f"Title: {title}\nURL: {url}\nDescription: {description}\n"

        markdown = res.get("markdown")
        markdown = markdown.strip() if isinstance(markdown, str) else ""
        if scrape_content and markdown and total_content_chars > 0:
            cap = min(per_result_chars, total_content_chars)
            if len(markdown) > cap:
                markdown = markdown[:cap] + "\n[... truncated ...]"
            total_content_chars -= min(len(markdown), cap)
            results_string += f"Content:\n{markdown}\n"

        results_string += "\n"

    return results_string or "No results found on Firecrawl search."


def firecrawl_scrape(url: str, only_main_content: bool = True, max_chars: int = 20000) -> str:
    """Extract the content of a webpage or PDF as markdown using the Firecrawl Scrape API.

    Uses the Firecrawl Scrape API (https://docs.firecrawl.dev/features/scrape). Handles JavaScript-rendered
    pages and PDF URLs in one call. extract_url_content remains the lighter choice for simple static HTML.

    Works without an API key (keyless tier, capped per IP per day). Set FIRECRAWL_API_KEY for higher limits.

    Args:
        url (str): Webpage or PDF URL to extract content from
        only_main_content (bool): Drop navigation, headers, footers and sidebars (default: True)
        max_chars (int): Truncate the returned markdown to this many characters. 0 disables truncation
            (default: 20000)

    Returns:
        str: Page content as markdown, or an error message

    """
    if not isinstance(url, str) or not url.strip():
        return "Error: url must be a non-empty string."
    try:
        max_chars = max(0, int(max_chars))
    except (TypeError, ValueError):
        return f"Error: max_chars must be an integer, got {max_chars!r}."

    payload = {"url": url, "formats": ["markdown"], "onlyMainContent": bool(only_main_content)}

    body = _firecrawl_request("scrape", payload)
    if isinstance(body, str):
        return f"Error scraping {url}: {body}"

    data = body.get("data")
    data = data if isinstance(data, dict) else {}
    metadata = data.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    status = metadata.get("statusCode")
    if isinstance(status, int) and status >= 400:
        return f"Error scraping {url}: the page returned HTTP {status}."

    markdown = data.get("markdown")
    markdown = markdown.strip() if isinstance(markdown, str) else ""
    if not markdown:
        return f"Firecrawl returned no text content for {url}."

    if max_chars and len(markdown) > max_chars:
        markdown = markdown[:max_chars] + "\n[... truncated ...]"

    return markdown
