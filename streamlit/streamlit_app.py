import gzip
import io
import logging
import os
import re
import shutil
import sys
import glob
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import streamlit as st
from PIL import Image
from langchain_core.messages import AIMessage, HumanMessage

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omics_horizon_app import (
    BIOMNI_DATA_PATH,
    CURRENT_ABS_DIR,
    GLOBAL_CSS_TEMPLATE,
    LLM_MODEL,
    TRANSLATIONS,
    WORKSPACE_PATH,
    ensure_session_defaults,
    setup_file_logger,
)
from omics_horizon_app.agent_runtime import (
    add_chat_message,
    build_agent_input_from_history,
    display_chat_files,
    format_agent_output_for_display,
    maybe_add_assistant_message,
    parse_step_progress,
    process_with_agent,
)
from omics_horizon_app.ui import render_control_panel
from omics_horizon_app.ui.data_panels import render_primary_panels
from omics_horizon_app.resources import load_logo_base64, LOGO_COLOR_PATH, LOGO_MONO_PATH
from omics_horizon_app.agent_service import get_or_create_agent


log = setup_file_logger("omics.streamlit_app", "streamlit_app.log")


def _display_workspace_path(path: Optional[str]) -> str:
    """Return workspace path relative to configured workspace root for display."""
    if not path:
        return "Not initialized"
    prefix = f"{WORKSPACE_PATH.rstrip(os.sep)}{os.sep}"
    if path == WORKSPACE_PATH:
        return "."
    if path.startswith(prefix):
        relative = path[len(prefix) :]
        return relative or "."
    try:
        rel_path = os.path.relpath(path, WORKSPACE_PATH)
        return rel_path if rel_path != "." else "."
    except Exception:
        return path


# Constants for data processing
MAX_DATA_COLUMNS_TO_SHOW = 20
MAX_SAMPLE_EXAMPLES = 5
MIN_COLUMN_PATTERN_LENGTH = 3
MAX_CONTENT_LENGTH_FOR_LLM = 15000
MAX_DISPLAY_TEXT_LENGTH = 8000
MIN_MEANINGFUL_CONTENT_LENGTH = 50
CHAT_ATTACHMENT_PATTERNS: tuple[str, ...] = (
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.bmp",
)


def _collect_workspace_artifacts(patterns: Iterable[str]) -> set[str]:
    """Return absolute paths for matching files in the session workspace."""
    workspace = st.session_state.get("work_dir")
    if not workspace:
        return set()

    collected: set[str] = set()
    for pattern in patterns:
        for path in glob.glob(os.path.join(workspace, pattern)):
            collected.add(os.path.abspath(path))
    return collected

@st.cache_data
def _get_logo_assets() -> tuple[str, str]:
    """Return cached base64 logos for light/dark themes."""
    return load_logo_base64(LOGO_COLOR_PATH), load_logo_base64(LOGO_MONO_PATH)


def _apply_global_theme(from_lims: bool) -> None:
    """Set page configuration and scoped CSS rules."""
    if not from_lims:
        st.set_page_config(
            page_title="OmicsHorizon™-Transcriptome",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    sidebar_rule = (
        '[data-testid="stSidebar"] {\n'
        "    min-width: 420px !important;\n"
        "    max-width: 420px !important;\n"
        "}\n"
        if not from_lims
        else ""
    )
    st.markdown(
        GLOBAL_CSS_TEMPLATE.format(sidebar_width_rule=sidebar_rule),
        unsafe_allow_html=True,
    )


def _render_sidebar_header() -> None:
    """Render OmicsHorizon logo header inside the sidebar."""
    color_logo, mono_logo = _get_logo_assets()
    with st.sidebar:
        if color_logo or mono_logo:
            st.markdown(
                f"""
            <div class="logo-container">
                <img src="data:image/svg+xml;base64,{color_logo}"
                     class="logo-light" alt="OMICS-HORIZON Logo">
                <img src="data:image/svg+xml;base64,{mono_logo}"
                     class="logo-dark" alt="OMICS-HORIZON Logo">
            </div>
            """,
                unsafe_allow_html=True,
            )
        st.markdown("---")


# Helper function for translations
def t(key):
    """Get translated text based on current language setting."""
    lang = st.session_state.get("language", "en")
    return TRANSLATIONS[lang].get(key, key)


def initialize_app_context(from_lims: bool, workspace_path: Optional[str]) -> None:
    """Configure Streamlit session, theme, and agent for the app."""
    ensure_session_defaults(from_lims=from_lims, workspace_path=workspace_path)
    st.session_state.agent = get_or_create_agent(log)
    _apply_global_theme(from_lims)
    _render_sidebar_header()


def save_uploaded_file(uploaded_file):
    """Save uploaded file to work directory."""
    file_path = os.path.join(st.session_state.work_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    log.info("Uploaded file saved: %s (%d bytes)", file_path, len(uploaded_file.getbuffer()))
    return uploaded_file.name


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    try:
        # Try PyMuPDF first (faster and better)
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except ImportError:
        try:
            # Fallback to pdfplumber
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                return "\n".join([page.extract_text() or "" for page in pdf.pages])
        except ImportError:
            # Last resort: use pypdf
            from pypdf import PdfReader

            reader = PdfReader(pdf_path)
            return "\n".join([page.extract_text() for page in reader.pages])


def find_section(
    text, section_name, start_keywords, end_keywords, max_chars=MAX_DISPLAY_TEXT_LENGTH
):
    """Generic function to find and extract a section from paper text.

    Args:
        text: Full paper text
        section_name: Name of section for logging
        start_keywords: List of keywords that indicate section start
        end_keywords: List of keywords that indicate section end
        max_chars: Maximum characters to return

    Returns:
        Extracted section text
    """
    section = ""
    lines = text.split("\n")
    in_section = False

    for line in lines:
        line_lower = line.lower().strip()

        # Start capturing at target section
        if any(keyword in line_lower for keyword in start_keywords):
            in_section = True
            continue

        # Stop at next major section
        if in_section and any(keyword in line_lower for keyword in end_keywords):
            break

        if in_section:
            section += line + "\n"

    # If section not found or too short, return truncated full text
    MIN_SECTION_LENGTH = 500
    if not section or len(section) < MIN_SECTION_LENGTH:
        return text[:max_chars]
    else:
        return section[:max_chars]


def find_methods_section(text):
    """Find and extract the Methods section from paper text."""
    methods_keywords = [
        "materials and methods",
        "methods",
        "materials & methods",
        "experimental procedures",
        "methodology",
        "experimental design",
    ]
    end_keywords = [
        "results",
        "discussion",
        "conclusion",
        "references",
        "acknowledgment",
        "supplementary",
        "data availability",
    ]

    return find_section(
        text,
        "Methods",
        methods_keywords,
        end_keywords,
        max_chars=MAX_DISPLAY_TEXT_LENGTH,
    )


def find_results_section(text):
    """Find and extract the Results section from paper text."""
    results_keywords = ["results", "results and discussion", "findings"]
    end_keywords = [
        "discussion",
        "conclusion",
        "materials and methods",
        "methods",
        "references",
        "acknowledgment",
        "supplementary",
    ]

    return find_section(
        text,
        "Results",
        results_keywords,
        end_keywords,
        max_chars=MAX_DISPLAY_TEXT_LENGTH,
    )


def smart_column_summary(
    columns,
    max_data_cols=MAX_DATA_COLUMNS_TO_SHOW,
    max_sample_examples=MAX_SAMPLE_EXAMPLES,
):
    """Intelligently summarize column names, distinguishing data columns from sample IDs.

    Args:
        columns: List of column names
        max_data_cols: Max number of data columns to show in full
        max_sample_examples: Number of sample examples to show

    Returns:
        Formatted string with smart column summary
    """
    if len(columns) <= max_data_cols:
        # Few columns - list all
        return "**All Columns:**\n" + "\n".join([f"- {col}" for col in columns])

    # Try to identify data columns vs sample columns
    # Common patterns for data columns
    data_keywords = [
        "gene",
        "id",
        "symbol",
        "name",
        "chr",
        "start",
        "end",
        "strand",
        "length",
        "type",
        "description",
        "annotation",
        "ensemble",
    ]

    data_cols = []
    sample_cols = []

    for col in columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in data_keywords):
            data_cols.append(col)
        else:
            sample_cols.append(col)

    # If we couldn't distinguish, use simple heuristic: first N are data, rest are samples
    if len(data_cols) == 0 and len(columns) > MAX_DATA_COLUMNS_TO_SHOW:
        data_cols = columns[:10]
        sample_cols = columns[10:]

    result = []

    # Data columns - list all if reasonable
    if data_cols:
        result.append("**Data Columns:**")
        for col in data_cols[:max_data_cols]:
            result.append(f"- {col}")
        if len(data_cols) > max_data_cols:
            result.append(
                f"- ... and {len(data_cols) - max_data_cols} more data columns"
            )

    # Sample columns - detect pattern and summarize
    if sample_cols:
        result.append(f"\n**Sample Columns ({len(sample_cols)} samples):**")

        # Detect common pattern
        pattern = detect_column_pattern(sample_cols)
        if pattern:
            result.append(f"Pattern: {pattern}")

        # Show examples
        if len(sample_cols) <= max_sample_examples * 2:
            result.append(f"Samples: {', '.join(sample_cols)}")
        else:
            examples = (
                sample_cols[:max_sample_examples]
                + ["..."]
                + sample_cols[-max_sample_examples:]
            )
            result.append(f"Examples: {', '.join(examples)}")

    return "\n".join(result)


def detect_column_pattern(columns):
    """Detect common pattern in column names.

    Returns pattern description like 'TCGA-XX-XXXX' or None
    """
    if len(columns) < MIN_COLUMN_PATTERN_LENGTH:
        return None

    # Sample a few columns
    sample = columns[: min(10, len(columns))]

    # Try to find common pattern
    # Check for TCGA pattern
    if all(col.startswith("TCGA-") for col in sample):
        return "TCGA-XX-XXXX format (TCGA sample IDs)"

    # Check for other common patterns
    # Pattern: PREFIX-numbers
    if all(re.match(r"^[A-Z]+[-_]\d+", col) for col in sample):
        prefix = re.match(r"^([A-Z]+)[-_]", sample[0]).group(1)
        return f"{prefix}-### format"

    # Pattern: All start with same prefix
    common_prefix = os.path.commonprefix(sample)
    if len(common_prefix) >= MIN_COLUMN_PATTERN_LENGTH:
        return f"{common_prefix}* format"

    return None


def analyze_data_direct(file_paths):
    """Analyze data files directly with LLM, without using agent.

    Returns only essential file information for briefing.
    """
    file_info = []
    for path in file_paths:
        try:
            # Get basic file info without loading full data
            file_name = os.path.basename(path)

            # Check if file exists
            if not os.path.exists(path):
                file_info.append({"name": file_name, "error": "File not found"})
                continue

            file_size = os.path.getsize(path)

            # Try to peek at structure with encoding handling
            try:
                if path.endswith(".gz"):
                    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                        first_line = f.readline()
                        if not first_line:
                            raise ValueError("Empty file")
                        columns = (
                            first_line.strip().split("\t")
                            if "\t" in first_line
                            else first_line.strip().split(",")
                        )
                else:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        first_line = f.readline()
                        if not first_line:
                            raise ValueError("Empty file")
                        columns = (
                            first_line.strip().split("\t")
                            if "\t" in first_line
                            else first_line.strip().split(",")
                        )
            except UnicodeDecodeError:
                # Try with latin-1 encoding as fallback
                with open(path, "r", encoding="latin-1") as f:
                    first_line = f.readline()
                    columns = (
                        first_line.strip().split("\t")
                        if "\t" in first_line
                        else first_line.strip().split(",")
                    )

            # Smart column summary
            column_summary = smart_column_summary(columns)

            info = {
                "name": file_name,
                "size": (
                    f"{file_size / (1024*1024):.2f} MB"
                    if file_size > 1024 * 1024
                    else f"{file_size / 1024:.2f} KB"
                ),
                "columns": len(columns),
                "column_summary": column_summary,
            }
            file_info.append(info)
        except (IOError, OSError) as e:
            file_info.append(
                {
                    "name": os.path.basename(path),
                    "error": f"File access error: {str(e)}",
                }
            )
        except ValueError as e:
            file_info.append({"name": os.path.basename(path), "error": str(e)})
        except Exception as e:
            file_info.append(
                {"name": os.path.basename(path), "error": f"Unexpected error: {str(e)}"}
            )

    # Create concise summary
    summary = "Files uploaded:\n"
    for info in file_info:
        if "error" in info:
            summary += f"\n- {info['name']}: Error reading file"
        else:
            summary += f"\n- {info['name']}: {info['size']}, {info['columns']} columns"

    # Add detailed column information
    detailed_info = summary + "\n\nDetailed Column Information:\n"
    for info in file_info:
        if "column_summary" in info:
            detailed_info += f"\n{info['name']}:\n{info['column_summary']}\n"

    # Use LLM for brief analysis
    llm = st.session_state.agent.llm

    prompt = f"""Based on the uploaded files, provide a concise briefing.

{detailed_info}

Provide a brief analysis with these sections:
## Data Overview
[File types, sizes, structure summary]

## Column Names
**CRITICAL: Use the column information provided above.**
- For data columns: list each one exactly as shown
- For sample columns: use the pattern and example format provided (do NOT list all 600 samples)
- This information will be used for downstream analysis

Example format:
**Data Columns:**
- gene_id
- gene_name
- ...

**Sample Columns (N samples):**
Pattern: TCGA-XX-XXXX format
Examples: TCGA-A1-A0SB, TCGA-A1-A0SD, ..., TCGA-ZZ-ZZZZ

## Key Variables
[Brief description of important columns and what they represent]

## Recommendations
[2-3 suggested analysis steps]

Keep it concise. Use the smart column summary format provided - do NOT enumerate all sample IDs."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def extract_workflow_from_paper(pdf_path, mode="integrated"):
    """Extract analysis workflow from PDF using Results + Methods sections.

    Args:
        pdf_path: Path to PDF file
        mode: "integrated" (Results+Methods), "methods_only", or "results_only"

    Returns:
        Structured workflow as numbered list
    """
    # Step 1: Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Step 2: Extract relevant sections based on mode
    if mode == "integrated":
        results_section = find_results_section(text)
        methods_section = find_methods_section(text)

        # Use LLM to integrate both sections
        llm = st.session_state.agent.llm

        prompt = f"""You are analyzing a bioinformatics research paper to extract the data analysis workflow.

TASK: Extract the complete analysis workflow in the order it was actually performed.

RESULTS SECTION (shows the analysis sequence and rationale):
{results_section}

METHODS SECTION (shows technical details and parameters):
{methods_section}

INSTRUCTIONS:
1. Identify the sequence of analyses from the Results section (look for "first", "next", "then", "finally", etc.)
2. For each analysis step, find the corresponding technical details from the Methods section
3. Create a numbered workflow that combines: analysis order + purpose + technical details

OUTPUT FORMAT (numbered list):
1. [Analysis name]: [Brief description of what was done]
   - Tool: [Software/package with version if available]
   - Parameters: [Key parameters, thresholds, or settings]
   - Purpose: [Why this step was performed - from Results context]

2. [Next analysis step]...

EXAMPLE:
1. Quality Control: Assess sequencing data quality
   - Tool: FastQC v0.11.9
   - Parameters: Default settings, min quality score > 20
   - Purpose: Filter low-quality reads before downstream analysis

Keep each step concise but include all essential information. Focus on computational/statistical analyses, not wet-lab procedures."""

        response = llm.invoke([HumanMessage(content=prompt)])
        workflow = response.content.strip()

    elif mode == "methods_only":
        methods_section = find_methods_section(text)
        llm = st.session_state.agent.llm

        prompt = f"""From the following Methods section, extract the data analysis workflow as a numbered list.

For each step, include:
- What analysis was performed
- Which tool/package/software was used
- Key parameters or thresholds

Methods section:
{methods_section}

Analysis workflow (numbered list):"""

        response = llm.invoke([HumanMessage(content=prompt)])
        workflow = response.content.strip()

    elif mode == "results_only":
        results_section = find_results_section(text)
        llm = st.session_state.agent.llm

        prompt = f"""From the following Results section, extract the data analysis workflow sequence.

Focus on the order of analyses performed (not biological findings).

Results section:
{results_section}

Analysis workflow (numbered list with brief descriptions):"""

        response = llm.invoke([HumanMessage(content=prompt)])
        workflow = response.content.strip()

    # Clean up the response
    workflow = re.sub(r"^```.*?\n", "", workflow)
    workflow = re.sub(r"\n```$", "", workflow)

    return workflow


# Keep old function for backward compatibility (calls new function)
def extract_method_from_paper(pdf_path):
    """Legacy function - calls new integrated workflow extraction."""
    return extract_workflow_from_paper(pdf_path, mode="methods_only")


def extract_key_findings(result_text):
    """Extract and format key findings from result text."""
    # Try to find sections in the result
    sections = {"summary": "", "methods": "", "results": "", "visualizations": ""}

    # Simple section detection
    if "summary" in result_text.lower() or "overview" in result_text.lower():
        sections["summary"] = "✅ Analysis completed"

    return sections


def post_process_with_llm(raw_result):
    """Use LLM to clean up the result and extract only the analytical content."""
    # First try to extract solution content
    solution_match = re.search(
        r"<solution>(.*?)</solution>", raw_result, flags=re.DOTALL
    )
    if solution_match:
        content = solution_match.group(1).strip()
    else:
        # Try observation tags - get the last one only
        observation_matches = re.findall(
            r"<observation>(.*?)</observation>", raw_result, flags=re.DOTALL
        )
        if observation_matches:
            content = observation_matches[-1].strip()
        else:
            # Remove execute blocks
            content = re.sub(r"<execute>.*?</execute>", "", raw_result, flags=re.DOTALL)

    # Aggressive cleaning of common artifacts
    artifacts_to_remove = [
        r"\[✓\].*?\n",  # Plan checkmarks
        r"Plan Update:.*?\n",  # Plan updates
        r"Executing Step.*?\n",  # Step execution
        r"#!BASH.*?\n",  # Bash commands
        r"<execute>.*?</execute>",  # Execute blocks
        r"<observation>.*?</observation>",  # Observation blocks (already extracted)
        r"```[\s\S]*?```",  # Code blocks
        r"print\(.*?\)",  # Print statements
        r"ls\s+",  # ls commands
        r"\.pdf",  # PDF references in commands
    ]

    for pattern in artifacts_to_remove:
        content = re.sub(pattern, "", content, flags=re.DOTALL)

    # If content is already clean (no code patterns), return as is
    if not any(
        pattern in content
        for pattern in [
            "print(",
            "try:",
            "pd.",
            "import ",
            "def ",
            "```",
            "Plan",
            "[✓]",
            "#!BASH",
        ]
    ):
        return content.strip()

    # Limit input length to avoid token limits
    if len(content) > MAX_DISPLAY_TEXT_LENGTH:
        content = content[:MAX_DISPLAY_TEXT_LENGTH] + "\n...(truncated)"

    # Use LLM to clean it up with very specific instructions
    cleanup_prompt = f"""Below is output that contains analysis steps mixed with code/logs. Extract ONLY the numbered analysis steps.

Remove all:
- Code (bash, python, etc)
- Plan updates
- Execute tags
- Print statements

Keep only the numbered list of analysis steps.

Raw:
{content}

Cleaned numbered list:"""

    try:
        # Use the agent's LLM to clean up
        from biomni.config import default_config
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model=default_config.llm, temperature=0)
        response = llm.invoke([HumanMessage(content=cleanup_prompt)])

        cleaned = response.content.strip()

        # Remove any remaining solution tags
        cleaned = re.sub(r"</?solution>", "", cleaned)

        return cleaned
    except Exception as e:
        # Fallback to basic cleaning if LLM fails
        print(f"LLM cleanup failed: {e}")
        return clean_code_artifacts(content)


def extract_solution_content(result_text):
    """Extract content from <solution> tags, which contains the final formatted answer."""
    import re

    # Use LLM-based post-processing for cleaner results
    return post_process_with_llm(result_text)


def clean_code_artifacts(text):
    """Remove code blocks and code-like artifacts from text."""
    # Remove code blocks (```...```)
    text = re.sub(r"```[\s\S]*?```", "", text)

    # Remove lines that look like code
    lines = text.split("\n")
    cleaned_lines = []
    skip_next_lines = 0

    for i, line in enumerate(lines):
        if skip_next_lines > 0:
            skip_next_lines -= 1
            continue

        stripped = line.strip()

        # Skip empty lines at the start
        if not stripped and not cleaned_lines:
            continue

        # Code patterns to skip
        code_patterns = [
            r"^print\s*\(",
            r"^try\s*:",
            r"^except\s+",
            r"^if\s+.*:$",
            r"^for\s+.*:$",
            r"^def\s+\w+",
            r"^import\s+",
            r"^from\s+\w+\s+import",
            r"^\w+\s*=\s*pd\.",
            r"^\w+\s*=\s*np\.",
            r"exit\(\)",
            r"\.read_csv\(",
            r"\.to_csv\(",
            r"FileNotFoundError",
            r"^---\s+Step\s+\d+",
            r"^---\s+Loading",
            r"successfully\.$",  # "loaded successfully."
        ]

        # Check if line matches code patterns
        is_code = any(re.search(pattern, stripped) for pattern in code_patterns)

        # Also skip lines that are mostly code-like (have parentheses and dots)
        if not is_code and "(" in stripped and ")" in stripped:
            # Count non-alphabetic characters
            alpha_count = sum(c.isalpha() or c.isspace() for c in stripped)
            total_count = len(stripped)
            if total_count > 0 and alpha_count / total_count < 0.6:
                is_code = True

        if not is_code:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def parse_structured_sections(text):
    """Parse text into sections based on headers."""
    sections = {}
    current_section = "General"
    current_content = []

    lines = text.split("\n")

    for line in lines:
        stripped = line.strip()

        # Skip lines that look like code artifacts
        if any(
            pattern in stripped
            for pattern in [
                "print(",
                "try:",
                "except",
                "import ",
                "pd.",
                "np.",
                "exit()",
            ]
        ):
            continue

        # Detect headers (##, ###, etc)
        header_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if header_match:
            header_text = header_match.group(2).strip()

            # Skip headers that look like code comments
            if not any(
                pattern in header_text for pattern in ["Step", "Loading", "---"]
            ):
                # Save previous section
                if current_content:
                    content = "\n".join(current_content).strip()
                    if content:  # Only save non-empty content
                        sections[current_section] = content

                # Start new section
                current_section = header_text
                current_content = []
        else:
            if stripped:  # Only add non-empty lines
                current_content.append(line)

    # Save last section
    if current_content:
        content = "\n".join(current_content).strip()
        if content:
            sections[current_section] = content

    return sections


def extract_key_metrics(text):
    """Extract key metrics and numbers from text for dashboard display."""
    metrics = []

    # Pattern to find: "label: number unit" or "label: number"
    patterns = [
        r"(\w+(?:\s+\w+)*?):\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(\w+)?",  # "Samples: 721 rows"
        r"(\d+(?:,\d+)*(?:\.\d+)?)\s+(\w+(?:\s+\w+)*)",  # "721 samples"
    ]

    lines = text.split("\n")
    for line in lines:
        # Look for dimension info like "721 samples" or "Dimensions: 721 x 89"
        if (
            "dimension" in line.lower()
            or "sample" in line.lower()
            or "gene" in line.lower()
        ):
            # Extract numbers
            numbers = re.findall(r"\d+(?:,\d+)*", line)
            if numbers:
                metrics.append({"text": line.strip(), "numbers": numbers})

        # Look for file sizes
        if any(unit in line.lower() for unit in ["kb", "mb", "gb", "bytes"]):
            size_match = re.search(
                r"(\d+(?:\.\d+)?)\s*(KB|MB|GB|Bytes)", line, re.IGNORECASE
            )
            if size_match:
                metrics.append(
                    {
                        "text": line.strip(),
                        "size": f"{size_match.group(1)} {size_match.group(2)}",
                    }
                )

    return metrics


def parse_analysis_steps(method_text):
    """Parse analysis method into structured steps.

    Returns list of dicts with step number, title, and description.
    """
    steps = []
    lines = method_text.split("\n")

    current_step = None
    current_desc = []

    for line in lines:
        # Match numbered steps: "1.", "1)", "Step 1:", etc.
        step_match = re.match(r"^(\d+)[\.\)]\s*(.+)$", line.strip())
        if not step_match:
            step_match = re.match(
                r"^Step\s+(\d+):\s*(.+)$", line.strip(), re.IGNORECASE
            )

        if step_match:
            # Save previous step
            if current_step:
                current_step["description"] = "\n".join(current_desc).strip()
                steps.append(current_step)

            # Start new step
            step_num = int(step_match.group(1))
            step_title = step_match.group(2).strip()

            # Extract main title (before colon if exists)
            title_parts = step_title.split(":", 1)
            main_title = title_parts[0].strip()

            current_step = {
                "step_num": step_num,
                "title": main_title,
                "full_text": step_title,
            }
            current_desc = [step_title] if len(title_parts) > 1 else []
        elif current_step and line.strip():
            current_desc.append(line.strip())

    # Save last step
    if current_step:
        current_step["description"] = "\n".join(current_desc).strip()
        steps.append(current_step)

    return steps


def match_images_to_steps(image_paths, analysis_steps):
    """Match generated images to analysis steps based on filename patterns.

    Returns dict: {step_num: [image_paths]}
    """
    step_images = {step["step_num"]: [] for step in analysis_steps}
    unmatched_images = []

    for img_path in image_paths:
        filename = os.path.basename(img_path).lower()
        matched = False

        # Try to match by step number in filename
        step_match = re.search(r"step[_\s]*(\d+)", filename)
        if step_match:
            step_num = int(step_match.group(1))
            if step_num in step_images:
                step_images[step_num].append(img_path)
                matched = True

        # Try to match by keywords in filename and step titles
        if not matched:
            for step in analysis_steps:
                step_keywords = step["title"].lower().split()
                # Check for keyword matches
                keywords = [
                    "pca",
                    "heatmap",
                    "volcano",
                    "boxplot",
                    "scatter",
                    "histogram",
                    "barplot",
                    "cluster",
                    "deg",
                    "pathway",
                    "correlation",
                    "distribution",
                    "quality",
                    "qc",
                ]

                for keyword in keywords:
                    if keyword in filename and keyword in step["title"].lower():
                        step_images[step["step_num"]].append(img_path)
                        matched = True
                        break

                if matched:
                    break

        if not matched:
            unmatched_images.append(img_path)

    # Distribute unmatched images across steps
    if unmatched_images:
        for idx, img in enumerate(unmatched_images):
            # Try to distribute evenly, or put all at the end
            if len(analysis_steps) > 0:
                # Add to last step or distribute
                last_step = analysis_steps[-1]["step_num"]
                step_images[last_step].append(img)

    return step_images


def extract_observation_timeline(raw_output, analysis_steps):
    """Extract observations sequentially and map them to analysis steps.

    This parses the agent's execution flow (plan checkboxes + observations)
    to reconstruct which observations belong to which steps.

    Args:
        raw_output: Complete agent output with <execute> and <observation> tags
        analysis_steps: User-defined steps from Panel 2

    Returns:
        dict: {step_num: {'observations': [str], 'confidence': str}}
    """
    # Extract all observations with their positions
    observations = []
    for match in re.finditer(
        r"<observation>(.*?)</observation>", raw_output, re.DOTALL
    ):
        obs_content = match.group(1).strip()
        obs_start = match.start()
        observations.append({"content": obs_content, "position": obs_start})

    if not observations:
        return {
            s["step_num"]: {"observations": [], "confidence": "low"}
            for s in analysis_steps
        }

    # Find all plan checkbox updates to track progress
    plan_updates = []
    for match in re.finditer(
        r"^\s*(\d+)\.\s*\[([✓✗ ]|[✅❌⬜])\]\s*(.+?)(?:\s*\(.*?\))?$",
        raw_output,
        re.MULTILINE,
    ):
        step_num = int(match.group(1))
        status_char = match.group(2)
        title = match.group(3).strip()
        position = match.start()

        # Determine status
        if status_char in ["✓", "✅"]:
            status = "completed"
        elif status_char in ["✗", "❌"]:
            status = "failed"
        else:
            status = "pending"

        plan_updates.append(
            {
                "step_num": step_num,
                "status": status,
                "title": title,
                "position": position,
            }
        )

    # Map observations to steps based on temporal proximity to plan updates
    step_observations = {
        s["step_num"]: {"observations": [], "confidence": "low"} for s in analysis_steps
    }

    if not plan_updates:
        # No plan updates found - distribute observations evenly
        obs_per_step = max(1, len(observations) // len(analysis_steps))
        for idx, step in enumerate(analysis_steps):
            start_idx = idx * obs_per_step
            end_idx = (
                start_idx + obs_per_step
                if idx < len(analysis_steps) - 1
                else len(observations)
            )
            step_observations[step["step_num"]]["observations"] = [
                obs["content"] for obs in observations[start_idx:end_idx]
            ]
            step_observations[step["step_num"]]["confidence"] = "low"
        return step_observations

    # Strategy: Assign observations to the step based on the nearest completed checkbox before it
    current_step = None

    # Sort plan updates by position
    plan_updates.sort(key=lambda x: x["position"])

    for obs in observations:
        obs_pos = obs["position"]

        # Find the most recent completed step before this observation
        recent_completed = None
        for update in reversed(plan_updates):
            if update["position"] < obs_pos and update["status"] == "completed":
                recent_completed = update["step_num"]
                break

        # If found, assign to that step
        if recent_completed and recent_completed in step_observations:
            step_observations[recent_completed]["observations"].append(obs["content"])
            step_observations[recent_completed]["confidence"] = "medium"
        elif len(analysis_steps) > 0:
            # Fallback: assign to first step
            first_step = analysis_steps[0]["step_num"]
            step_observations[first_step]["observations"].append(obs["content"])

    return step_observations


def _assign_images_by_time(step_data, analysis_steps, all_images):
    """Assign unmatched images to steps based on creation time order.

    Assumes images are created in the order of analysis steps.
    """
    # Collect all images already assigned
    assigned_images = set()
    for data in step_data.values():
        for img in data.get("images", []):
            assigned_images.add(img)

    # Get unassigned images in creation time order
    unassigned_images = [img for img in all_images if img not in assigned_images]

    if not unassigned_images:
        return step_data

    # Distribute unassigned images across steps
    # Strategy: if we have N steps and M unassigned images, distribute evenly
    num_steps = len(analysis_steps)
    images_per_step = max(1, len(unassigned_images) // num_steps)

    img_idx = 0
    for step in analysis_steps:
        step_num = step["step_num"]

        # If this step has no images yet, assign some
        if not step_data[step_num].get("images"):
            # Take next batch of images
            batch_size = min(images_per_step, len(unassigned_images) - img_idx)
            if batch_size > 0:
                step_data[step_num]["images"] = unassigned_images[
                    img_idx : img_idx + batch_size
                ]
                img_idx += batch_size

    # Assign any remaining images to the last step
    if img_idx < len(unassigned_images):
        last_step_num = analysis_steps[-1]["step_num"]
        step_data[last_step_num]["images"].extend(unassigned_images[img_idx:])

    return step_data


def reorganize_results_by_steps_with_llm(raw_output, analysis_steps, all_images):
    """Use LLM to reorganize agent output according to analysis steps.

    Args:
        raw_output: Complete agent output including all execution details
        analysis_steps: Steps from Panel 2
        all_images: List of generated image paths

    Returns:
        dict: {step_num: {'results': str, 'images': [paths], 'confidence': str}}
    """
    # Extract solution content first
    solution_match = re.search(
        r"<solution>(.*?)</solution>", raw_output, flags=re.DOTALL
    )
    if solution_match:
        main_content = solution_match.group(1).strip()
    else:
        # Try to get observation tags
        observation_matches = re.findall(
            r"<observation>(.*?)</observation>", raw_output, flags=re.DOTALL
        )
        if observation_matches:
            main_content = "\n\n".join(observation_matches)
        else:
            main_content = raw_output

    # First, try to parse using our structured markers (=== Step X: ... ===)
    step_data = {}
    structured_parse_success = False

    for step in analysis_steps:
        step_num = step["step_num"]

        # Try to find structured markers
        marker_pattern = rf"===\s*Step\s+{step_num}[:\s]+(.*?)===\s*(.*?)\s*===\s*End\s+Step\s+{step_num}\s*==="
        match = re.search(marker_pattern, main_content, re.DOTALL | re.IGNORECASE)

        if match:
            structured_parse_success = True
            step_title = match.group(1).strip()
            step_content = match.group(2).strip()

            # Extract image filenames from content
            image_files = re.findall(
                r"[\w\-_]+\.(?:png|jpg|jpeg|gif|bmp)", step_content, re.IGNORECASE
            )

            step_images = []
            for fname in image_files:
                for img_path in all_images:
                    if os.path.basename(img_path) == fname:
                        step_images.append(img_path)
                        break

            step_data[step_num] = {
                "results": step_content if len(step_content) > 10 else None,
                "images": step_images,
                "confidence": "high",
            }
        else:
            step_data[step_num] = {"results": None, "images": [], "confidence": "low"}

    # If structured parsing worked for most steps, return it
    successful_steps = sum(1 for s in step_data.values() if s["confidence"] == "high")
    if successful_steps >= len(analysis_steps) * 0.5:  # At least 50% success
        # For steps without images, try to assign based on creation time
        step_data = _assign_images_by_time(step_data, analysis_steps, all_images)
        return step_data

    # Second fallback: Try observation timeline parsing
    # This extracts observations sequentially and maps them to steps based on plan execution
    observation_timeline = extract_observation_timeline(raw_output, analysis_steps)

    # Merge observation data with existing step_data
    for step_num, obs_data in observation_timeline.items():
        if obs_data["observations"] and len(obs_data["observations"]) > 0:
            # Combine all observations for this step
            combined_observations = "\n\n".join(obs_data["observations"])

            # If we don't have results yet, or existing results are low confidence, use observations
            if step_data[step_num]["confidence"] == "low" or not step_data[
                step_num
            ].get("results"):
                step_data[step_num]["results"] = combined_observations
                step_data[step_num]["confidence"] = obs_data["confidence"]
            else:
                # Append observations to existing results
                existing = step_data[step_num].get("results", "")
                step_data[step_num]["results"] = (
                    existing + "\n\n" + combined_observations
                )

    # Check if observation parsing gave us sufficient coverage
    successful_steps = sum(
        1 for s in step_data.values() if s.get("results") and len(s["results"]) > 20
    )
    if (
        successful_steps >= len(analysis_steps) * 0.3
    ):  # At least 30% have substantial content
        # Apply time-based image assignment
        step_data = _assign_images_by_time(step_data, analysis_steps, all_images)
        return step_data

    # Third fallback: LLM reorganization (last resort)

    # Limit content size
    if len(main_content) > MAX_CONTENT_LENGTH_FOR_LLM:
        main_content = (
            main_content[:MAX_CONTENT_LENGTH_FOR_LLM] + "\n...(truncated for length)"
        )

    # Create list of steps for prompt
    steps_list = "\n".join(
        [f"Step {s['step_num']}: {s['title']}" for s in analysis_steps]
    )

    # Create list of image filenames
    image_filenames = [os.path.basename(img) for img in all_images]
    images_list = "\n".join([f"- {fname}" for fname in image_filenames])

    prompt = f"""You are analyzing bioinformatics workflow results. The user has defined analysis steps from a paper, and an agent executed various analyses. Your task is to map the agent's results to the user's defined steps.

User's Defined Analysis Steps:
{steps_list}

Generated Image Files:
{images_list}

Agent's Complete Output:
{main_content}

TASK: For each step (1, 2, 3...), extract:
1. What results/findings correspond to this step (be specific, include numbers/statistics if available)
2. Which image file(s) relate to this step (by filename)

Format your response EXACTLY as follows:
STEP_1
RESULTS: [Detailed findings for step 1, or "No specific results found" if nothing matches]
IMAGES: [Comma-separated image filenames, or "None" if no images match]
END_STEP_1

STEP_2
RESULTS: [Detailed findings for step 2, or "No specific results found"]
IMAGES: [Comma-separated image filenames, or "None"]
END_STEP_2

... (continue for all steps)

Be thorough but concise. If you're uncertain about a mapping, say so."""

    try:
        llm = st.session_state.agent.llm
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed_response = response.content.strip()

        # Parse LLM response
        step_data = {}
        for step in analysis_steps:
            step_num = step["step_num"]

            # Find content for this step
            step_pattern = rf"STEP_{step_num}\s*\n\s*RESULTS:\s*(.*?)\s*IMAGES:\s*(.*?)\s*END_STEP_{step_num}"
            match = re.search(step_pattern, parsed_response, re.DOTALL | re.IGNORECASE)

            if match:
                results = match.group(1).strip()
                images_str = match.group(2).strip()

                # Parse image filenames
                step_images = []
                if images_str and images_str.lower() not in [
                    "none",
                    "no images",
                    "n/a",
                ]:
                    # Extract filenames
                    mentioned_files = re.findall(
                        r"[\w\-_]+\.(?:png|jpg|jpeg|gif|bmp)", images_str, re.IGNORECASE
                    )
                    for fname in mentioned_files:
                        # Find full path
                        for img_path in all_images:
                            if os.path.basename(img_path) == fname:
                                step_images.append(img_path)
                                break

                step_data[step_num] = {
                    "results": results if results and len(results) > 10 else None,
                    "images": step_images,
                    "confidence": "high" if match else "low",
                }
            else:
                step_data[step_num] = {
                    "results": None,
                    "images": [],
                    "confidence": "low",
                }

        # Apply time-based image assignment for missing images
        step_data = _assign_images_by_time(step_data, analysis_steps, all_images)

        return step_data

    except Exception as e:
        st.error(f"Error in LLM reorganization: {e}")
        # Fallback to empty data with time-based image assignment
        fallback_data = {
            s["step_num"]: {"results": None, "images": [], "confidence": "low"}
            for s in analysis_steps
        }
        fallback_data = _assign_images_by_time(
            fallback_data, analysis_steps, all_images
        )
        return fallback_data


def extract_by_simple_headers(text, analysis_steps):
    """Simple header-based extraction - most permissive approach.

    Looks for various header patterns without strict requirements.
    """
    step_results = {}

    for step in analysis_steps:
        step_num = step["step_num"]
        step_title = step["title"]

        # Try multiple patterns (from most specific to most general)
        patterns = [
            rf"===\s*Step\s+{step_num}[:\s]+.*?===(.*?)===\s*End\s+Step\s+{step_num}\s*===",
            rf"##\s+Step\s+{step_num}[:\s]+(.*?)(?=##\s+Step\s+\d+|$)",
            rf"#\s+Step\s+{step_num}[:\s]+(.*?)(?=#\s+Step\s+\d+|$)",
            rf"Step\s+{step_num}[:\s]+{re.escape(step_title)}(.*?)(?=Step\s+\d+|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if len(content) > MIN_MEANINGFUL_CONTENT_LENGTH:  # Meaningful content
                    step_results[step_num] = content
                    break

    return step_results


def run_omicshorizon_app(from_lims=False, workspace_path=None):
    """Main function to run OmicsHorizon Transcriptome Analysis App

    Args:
        from_lims: If True, app is launched from LIMS with pre-selected data files
        workspace_path: Path to workspace directory (used when from_lims=True)
    """

    initialize_app_context(from_lims=from_lims, workspace_path=workspace_path)
    st.session_state.from_lims = from_lims
    color_logo, mono_logo = _get_logo_assets()

    # Handle LIMS integration

    # Auto-load data from LIMS if available
    if (
        from_lims
        and "selected_data_files" in st.session_state
        and st.session_state.selected_data_files
    ):
        # Load pre-selected files from LIMS
        if not st.session_state.data_files:  # Only load if not already loaded
            st.session_state.data_files = []
            st.session_state.data_briefing = ""

            # Get file paths from workspace (copied by LIMS)
            for file_info in st.session_state.selected_data_files:
                filename = file_info["name"]
                file_path = os.path.join(st.session_state.work_dir, filename)

                if os.path.exists(file_path):
                    if filename not in st.session_state.data_files:
                        st.session_state.data_files.append(filename)

            # Auto-generate briefing for loaded files
            if st.session_state.data_files and not st.session_state.data_briefing:
                file_paths = [
                    os.path.join(st.session_state.work_dir, fname)
                    for fname in st.session_state.data_files
                ]
                try:
                    result = analyze_data_direct(file_paths)
                    st.session_state.data_briefing = result
                except Exception as e:
                    st.session_state.data_briefing = f"Error analyzing data: {str(e)}"

    # Main logo
    if color_logo or mono_logo:
        st.markdown(
            f"""
        <div style="text-align: center; margin-bottom: 1rem; line-height: 0;">
            <img src="data:image/svg+xml;base64,{color_logo}"
                 class="logo-light main-logo" alt="OMICS-HORIZON Logo"
                 style="max-width: 600px; height: auto; margin: 0 auto;">
            <img src="data:image/svg+xml;base64,{mono_logo}"
                 class="logo-dark main-logo" alt="OMICS-HORIZON Logo"
                 style="max-width: 600px; height: auto; margin: 0 auto;">
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Title section
    # st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem; margin-top: 0.5rem; margin-bottom: 2rem;">AI-Powered Transcriptomic Analysis Platform</p>', unsafe_allow_html=True)
    st.markdown("---")

    render_primary_panels(
        from_lims=from_lims,
        t=t,
        save_uploaded_file=save_uploaded_file,
        analyze_data_direct=analyze_data_direct,
        extract_workflow_from_paper=extract_workflow_from_paper,
    )

    st.markdown("---")

    # Check if ready to start
    if st.session_state.data_files and st.session_state.analysis_method:
        # Parse analysis steps
        analysis_steps = parse_analysis_steps(st.session_state.analysis_method)

        if analysis_steps:
            # Set default mode to sequential
            if "interaction_mode" not in st.session_state:
                st.session_state.interaction_mode = "sequential"

            # Always use sequential mode
            render_sequential_interactive_mode(analysis_steps)

        else:
            st.warning(
                "⚠️ Could not parse analysis steps from Panel 2. Please check the format."
            )

    elif not st.session_state.data_files:
        st.warning("⚠️ Please upload data files in Panel 1")
    elif not st.session_state.analysis_method:
        st.warning("⚠️ Please upload a paper or define analysis method in Panel 2")

    # Sidebar
    with st.sidebar:
        # Q&A Section at the very top
        st.markdown(f"### {t('qa_title')}")

        # Check if there's any analysis to ask about
        has_analysis = bool(
            st.session_state.steps_state
            and any(
                s.get("status") == "completed"
                for s in st.session_state.steps_state.values()
            )
        )

        if has_analysis:
            with st.expander(t("qa_ask_questions"), expanded=False):
                st.caption(t("qa_caption"))

                # Display chat history
                for idx, msg in enumerate(st.session_state.qa_history):
                    if msg["role"] == "user":
                        user_label = (
                            "🙋 당신:"
                            if st.session_state.language == "ko"
                            else "🙋 You:"
                        )
                        st.markdown(f"**{user_label}** {msg['content']}")
                    else:
                        assistant_label = (
                            "🤖 어시스턴트:"
                            if st.session_state.language == "ko"
                            else "🤖 Assistant:"
                        )
                        st.markdown(f"**{assistant_label}**\n\n{msg['content']}")

                    if idx < len(st.session_state.qa_history) - 1:
                        st.markdown("---")

                # Question input
                question = st.text_input(
                    "Your question:" if st.session_state.language == "en" else "질문:",
                    key="qa_input",
                    placeholder=t("qa_placeholder"),
                    label_visibility="collapsed",
                )

                col1, col2 = st.columns([3, 1])

                with col1:
                    ask_button_label = (
                        "🚀 Ask" if st.session_state.language == "ko" else "🚀 질문"
                    )
                    if st.button(
                        ask_button_label,
                        key="ask_button",
                        use_container_width=True,
                        type="primary",
                    ):
                        if question and question.strip():
                            # Add user question
                            st.session_state.qa_history.append(
                                {"role": "user", "content": question}
                            )

                            # Get answer
                            thinking_msg = (
                                "🤔 생각 중..."
                                if st.session_state.language == "ko"
                                else "🤔 Thinking..."
                            )
                            with st.spinner(thinking_msg):
                                answer = answer_qa_question(question)

                            # Add assistant answer
                            st.session_state.qa_history.append(
                                {"role": "assistant", "content": answer}
                            )

                            st.rerun()
                        else:
                            warning_msg = (
                                "질문을 입력하세요"
                                if st.session_state.language == "ko"
                                else "Please enter a question"
                            )
                            st.warning(warning_msg)

                with col2:
                    clear_label = (
                        "🗑️ 지우기" if st.session_state.language == "ko" else "🗑️ Clear"
                    )
                    if st.button(clear_label, key="clear_qa", use_container_width=True):
                        st.session_state.qa_history = []
                        st.rerun()

                # Show helpful prompts
                if not st.session_state.qa_history:
                    st.markdown("---")
                    if st.session_state.language == "ko":
                        st.caption("**예시 질문:**")
                        st.caption("• Step 2의 주요 발견은 무엇인가요?")
                        st.caption("• 왜 이 통계 검정을 선택했나요?")
                        st.caption("• volcano plot을 설명해주세요")
                        st.caption("• 이 p-value는 무엇을 의미하나요?")
                    else:
                        st.caption("**Example questions:**")
                        st.caption("• What were the main findings in Step 2?")
                        st.caption("• Why was this statistical test chosen?")
                        st.caption("• Can you explain the volcano plot?")
                        st.caption("• What do these p-values indicate?")
        else:
            st.info(t("qa_no_analysis"))

        st.markdown("---")

        # Control Panel and Session Info
        st.markdown(f"## {t('control_panel')}")

        st.markdown(f"### {t('session_info')}")
        workspace_display = _display_workspace_path(st.session_state.work_dir)
        st.info(
            f"""
        - Data files: {len(st.session_state.data_files)}
        - Paper files: {len(st.session_state.paper_files)}
        - Method defined: {'✅' if st.session_state.analysis_method else '❌'}
        - Work directory: `{workspace_display}`
        """
        )

        st.markdown("---")

        # Reset Analysis button
        if st.session_state.steps_state:
            if st.button(
                "🔄 Reset Analysis", key="reset_analysis", use_container_width=True
            ):
                st.session_state.steps_state = {}
                st.session_state.current_step = 0
                reset_msg = (
                    "✅ Analysis reset!"
                    if st.session_state.language == "en"
                    else "✅ 분석이 초기화되었습니다!"
                )
                st.success(reset_msg)
                st.rerun()
            st.markdown("---")

        # Clear all button
        if st.button(t("clear_all"), key="clear_all", use_container_width=True):
            st.session_state.data_files = []
            st.session_state.data_briefing = ""
            st.session_state.paper_files = []
            st.session_state.analysis_method = ""
            st.session_state.steps_state = {}
            st.session_state.current_step = 0
            st.session_state.qa_history = []
            st.session_state.message_history = []
            success_msg = (
                "✅ All data cleared!"
                if st.session_state.language == "en"
                else "✅ 모든 데이터가 삭제되었습니다!"
            )
            st.success(success_msg)
            st.rerun()

        st.markdown("---")

        # Instructions
        with st.expander(t("instructions")):
            st.markdown(
                """
            ### How to use (Interactive Mode):

            1. **Upload Data** (Panel 1)
               - Upload CSV, Excel, or other data files
               - Click "Analyze Data" to get a briefing

            2. **Upload Paper** (Panel 2)
               - Upload a research paper (PDF)
               - Click "Extract Analysis Workflow"
               - Edit the workflow if needed

            3. **Interactive Step-by-Step Analysis** (Panel 3)
               - **Start Step 1**: Click "▶️ Start Step 1" button
               - **Review Results**: Check the analysis output and figures
               - **Provide Feedback** (optional): Enter modifications in natural language
               - **Re-run** (if needed): Click "🔄 Re-run" to apply feedback
               - **Next Step**: Click "▶️ Next" to proceed to Step 2
               - **Repeat** for all steps

            ### Tips:
            - Each step builds on previous results
            - You can re-run any step with different parameters
            - Use natural language for feedback (e.g., "Use p-value < 0.01")
            - All results are automatically saved
            """
            )

        st.markdown("---")
        st.markdown("### 🔧 Settings")
        st.text(f"Model: {LLM_MODEL}")
        # st.text(f"Path: {BIOMNI_DATA_PATH}")

        st.markdown("---")

        # Language selector at the bottom
        st.markdown(f"### {t('language')}")
        col_en, col_ko = st.columns(2)
        with col_en:
            if st.button(
                "English",
                key="lang_en",
                use_container_width=True,
                type="primary" if st.session_state.language == "en" else "secondary",
            ):
                st.session_state.language = "en"
                st.rerun()
        with col_ko:
            if st.button(
                "한국어",
                key="lang_ko",
                use_container_width=True,
                type="primary" if st.session_state.language == "ko" else "secondary",
            ):
                st.session_state.language = "ko"
                st.rerun()


# =============================================================================
# INTERACTIVE MODE: STEP-BY-STEP EXECUTION
# =============================================================================


def initialize_step_state(step_num, step_info):
    """Initialize state for a single step"""
    if step_num not in st.session_state.steps_state:
        st.session_state.steps_state[step_num] = {
            "status": "pending",  # 'pending', 'in_progress', 'completed', 'error'
            "title": step_info["title"],
            "description": step_info.get("description", ""),
            "result": None,  # Raw agent output
            "solution": None,  # Extracted solution content (clean)
            "formatted_process": None,  # Fully formatted process for expander
            "files": [],
            "feedback": None,
            "iteration": 0,
        }


def get_previous_context(step_num):
    """Get context from previous steps to pass to current step"""
    if step_num == 1:
        return ""

    context_parts = []

    for i in range(1, step_num):
        if i in st.session_state.steps_state:
            step_data = st.session_state.steps_state[i]

            if step_data["status"] == "completed" and step_data["result"]:
                # Extract key information from previous step
                context_parts.append(f"=== Previous Step {i}: {step_data['title']} ===")

                # Extract observations
                observations = re.findall(
                    r"<observation>(.*?)</observation>", step_data["result"], re.DOTALL
                )

                if observations:
                    # Use last observation (usually the most relevant)
                    last_obs = observations[-1].strip()
                    # Truncate if too long
                    if len(last_obs) > 500:
                        last_obs = last_obs[:500] + "... (truncated)"
                    context_parts.append(f"Key findings: {last_obs}")

                # Include generated files
                if step_data["files"]:
                    file_list = ", ".join(
                        [os.path.basename(f) for f in step_data["files"]]
                    )
                    context_parts.append(f"Generated files: {file_list}")

                context_parts.append("")  # Empty line

    return "\n".join(context_parts)


def get_available_result_files(step_num):
    """Get all result files available from previous steps"""
    import os
    import glob
    from pathlib import Path

    workspace = st.session_state.work_dir
    available_files = []

    # Get all files in workspace
    all_files = []
    for ext in [
        "*.csv",
        "*.txt",
        "*.tsv",
        "*.xlsx",
        "*.xls",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.pdf",
    ]:
        all_files.extend(glob.glob(os.path.join(workspace, ext)))

    # Original data files (exclude these from result files)
    original_files = (
        set(st.session_state.data_files) if st.session_state.data_files else set()
    )

    # Categorize files by creation time and step association
    result_files = []
    for file_path in all_files:
        filename = os.path.basename(file_path)

        # Skip original data files
        if filename in original_files:
            continue

        # Get file info
        try:
            stat = os.stat(file_path)
            created_time = stat.st_ctime

            # Try to associate with step based on filename or creation time
            associated_step = None
            if (
                f"step{step_num-1}" in filename.lower()
                or f"step_{step_num-1}" in filename.lower()
            ):
                associated_step = step_num - 1
            elif "step" in filename.lower():
                # Extract step number from filename
                import re

                step_match = re.search(r"step[_\s]?(\d+)", filename.lower())
                if step_match:
                    associated_step = int(step_match.group(1))

            file_info = {
                "path": file_path,
                "name": filename,
                "size": stat.st_size,
                "created": created_time,
                "associated_step": associated_step,
                "extension": Path(filename).suffix.lower(),
            }

            # Only include files created before this step execution
            # (approximate by checking if they exist and are not from future steps)
            if associated_step is None or associated_step < step_num:
                result_files.append(file_info)

        except OSError:
            continue

    # Sort by creation time (newest first)
    result_files.sort(key=lambda x: x["created"], reverse=True)

    return result_files


def execute_single_step(step_num, step_info):
    """Execute a single analysis step"""

    # Initialize step state if not exists
    if step_num not in st.session_state.steps_state:
        st.session_state.steps_state[step_num] = {
            "status": "pending",
            "title": step_info["title"],
            "description": step_info.get("description", ""),
            "result": None,
            "solution": None,
            "formatted_process": None,
            "files": [],
            "feedback": None,
            "iteration": 0,
        }

    # Update status
    st.session_state.steps_state[step_num]["status"] = "in_progress"
    st.session_state.steps_state[step_num]["iteration"] += 1

    # Get previous context
    previous_context = get_previous_context(step_num)

    # Get available result files from previous steps
    available_result_files = get_available_result_files(step_num)

    # Get feedback if this is a re-run
    feedback = st.session_state.steps_state[step_num].get("feedback")

    # Build prompt
    data_info = ", ".join([f"`{f}`" for f in st.session_state.data_files])

    # Add available result files to prompt
    result_files_info = ""
    if available_result_files:
        result_files_info = "\n\nAVAILABLE RESULT FILES FROM PREVIOUS STEPS:"
        for file_info in available_result_files[:10]:  # Limit to 10 most recent
            result_files_info += f"\n- {file_info['name']} ({file_info['extension']}, {file_info['size']/1024:.1f} KB)"
            if file_info["associated_step"]:
                result_files_info += f" - from Step {file_info['associated_step']}"

    prompt = f"""Perform Step {step_num} of the bioinformatics analysis.

DATA FILES: {data_info}

DATA BRIEFING:
{st.session_state.data_briefing if st.session_state.data_briefing else "Files are available in the working directory"}

⚠️ CRITICAL - COLUMN NAME VERIFICATION:
Before accessing any columns:
1. Run: print("Available columns:", df.columns.tolist())
2. Use df.columns to get actual column names


Description: {step_info.get('full_text', '')}

INSTRUCTIONS:
- Execute this step thoroughly
- **IMPORTANT**: Check and utilize result files from previous steps when relevant
- Load any CSV/TSV/TXT files from previous steps if they contain processed data you need
- Reference previous analysis results to build upon existing work
- Save any plots with descriptive filenames (e.g., "step{step_num}_*.png")
- Provide detailed results in <solution> tag
- Include specific numbers and statistics
"""

    # Execute with agent
    try:
        log.info("Executing step %s - %s", step_num, step_info.get("title", ""))
        result = process_with_agent(prompt, show_process=True, use_history=False)
        log.info("Step %s execution finished", step_num)

        # Get generated files (images created during this step)
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"]
        all_images = []
        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(st.session_state.work_dir, ext)))

        # Filter to new files (created after this step started)
        new_files = [f for f in all_images if f not in get_all_previous_files(step_num)]

        # Extract solution content (clean results without execution details)
        solution_match = re.search(r"<solution>(.*?)</solution>", result, re.DOTALL)
        if solution_match:
            solution_content = solution_match.group(1).strip()

            # AGGRESSIVE CLEANING: Remove all execution artifacts from solution

            # 1. Remove XML tags
            solution_content = re.sub(
                r"<execute>.*?</execute>", "", solution_content, flags=re.DOTALL
            )
            solution_content = re.sub(
                r"<observation>.*?</observation>", "", solution_content, flags=re.DOTALL
            )
            solution_content = re.sub(
                r"<think>.*?</think>", "", solution_content, flags=re.DOTALL
            )

            # 2. Remove ALL code blocks (they should be in process, not results)
            solution_content = re.sub(
                r"```[a-z]*\n.*?```", "", solution_content, flags=re.DOTALL
            )

            # 3. Remove plan checkboxes and markers
            solution_content = re.sub(
                r"^\s*\d+\.\s*\[[\s✓✗✅❌⬜]\].*?$",
                "",
                solution_content,
                flags=re.MULTILINE,
            )
            solution_content = re.sub(r"===.*?===", "", solution_content)
            solution_content = re.sub(r"Plan Update:.*?\n", "", solution_content)

            # 4. Remove code execution indicators
            solution_content = re.sub(
                r"🐍\s*\*\*코드 실행.*?\*\*", "", solution_content
            )
            solution_content = re.sub(
                r"📊\s*\*\*코드 실행.*?\*\*", "", solution_content
            )
            solution_content = re.sub(
                r"🔧\s*\*\*코드 실행.*?\*\*", "", solution_content
            )
            solution_content = re.sub(
                r"✅\s*\*\*실행 성공.*?\*\*", "", solution_content
            )
            solution_content = re.sub(
                r"❌\s*\*\*실행 오류.*?\*\*", "", solution_content
            )

            # 5. Remove horizontal rules (often used as separators in process)
            solution_content = re.sub(
                r"^---+$", "", solution_content, flags=re.MULTILINE
            )

            # 6. Remove multiple blank lines
            solution_content = re.sub(r"\n{3,}", "\n\n", solution_content)
            solution_content = solution_content.strip()

            # 7. If solution is now empty or too short, provide a message
            if not solution_content or len(solution_content) < 20:
                solution_content = "✅ Analysis completed successfully.\n\nPlease see 'View Analysis Process' below for detailed execution steps and 'Figures' section for generated visualizations."
        else:
            # Fallback: use last observation
            observations = re.findall(
                r"<observation>(.*?)</observation>", result, re.DOTALL
            )
            solution_content = (
                observations[-1].strip()
                if observations
                else "Analysis completed. See process details below."
            )

        # Update step state
        st.session_state.steps_state[step_num]["status"] = "completed"
        st.session_state.steps_state[step_num]["result"] = result  # Raw result
        st.session_state.steps_state[step_num][
            "solution"
        ] = solution_content  # Clean solution only
        st.session_state.steps_state[step_num]["formatted_process"] = (
            format_agent_output_for_display(result)
        )  # Full formatted process
        st.session_state.steps_state[step_num]["files"] = new_files
        st.session_state.steps_state[step_num][
            "feedback"
        ] = None  # Clear feedback after execution

        # Don't add chat message here - it will be handled by render_sequential_interactive_mode

        # Update current step
        st.session_state.current_step = step_num

        return True

    except Exception as e:
        st.session_state.steps_state[step_num]["status"] = "error"
        st.session_state.steps_state[step_num]["result"] = f"Error: {str(e)}"
        return False


def get_all_previous_files(step_num):
    """Get all files generated in previous steps"""
    all_files = []
    for i in range(1, step_num):
        if i in st.session_state.steps_state:
            all_files.extend(st.session_state.steps_state[i].get("files", []))
    return all_files


def get_qa_context():
    """Get context from all completed steps for Q&A"""
    if not st.session_state.steps_state:
        return "No analysis has been performed yet."

    context_parts = []

    # Add data briefing if available
    if st.session_state.data_briefing:
        context_parts.append("=== DATA BRIEFING ===")
        context_parts.append(st.session_state.data_briefing[:1000])
        context_parts.append("")

    # Add analysis method if available
    if st.session_state.analysis_method:
        context_parts.append("=== ANALYSIS WORKFLOW ===")
        context_parts.append(st.session_state.analysis_method[:1000])
        context_parts.append("")

    # Add completed steps
    context_parts.append("=== COMPLETED ANALYSIS STEPS ===")

    for step_num in sorted(st.session_state.steps_state.keys()):
        step_data = st.session_state.steps_state[step_num]

        if step_data["status"] == "completed":
            context_parts.append(f"\n--- Step {step_num}: {step_data['title']} ---")

            # Extract key observations
            if step_data["result"]:
                observations = re.findall(
                    r"<observation>(.*?)</observation>", step_data["result"], re.DOTALL
                )

                if observations:
                    # Use last 2 observations (most recent)
                    for obs in observations[-2:]:
                        truncated = obs.strip()[:800]
                        context_parts.append(f"Results: {truncated}...")

            # List generated files
            if step_data["files"]:
                file_names = [os.path.basename(f) for f in step_data["files"]]
                context_parts.append(f"Generated files: {', '.join(file_names)}")

            context_parts.append("")

    full_context = "\n".join(context_parts)

    # Limit total context length
    max_context_length = 8000
    if len(full_context) > max_context_length:
        full_context = (
            full_context[:max_context_length] + "\n\n... (context truncated for length)"
        )

    return full_context


def render_batch_interactive_mode(analysis_steps):
    """Render the traditional batch mode where all steps are visible"""
    # Summary bar
    total_steps = len(analysis_steps)
    completed_steps = sum(
        1
        for s in st.session_state.steps_state.values()
        if s.get("status") == "completed"
    )

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        progress = completed_steps / total_steps if total_steps > 0 else 0
        st.progress(progress)
        st.caption(
            f"Progress: {completed_steps}/{total_steps} steps completed ({int(progress * 100)}%)"
        )

    with col2:
        st.metric("Total Steps", total_steps)

    with col3:
        st.metric("Completed", completed_steps)

    st.markdown("---")

    # Info banner
    st.info(t("batch_mode_desc"))

    st.markdown("---")

    # Render each step
    for step in analysis_steps:
        render_step_panel(step["step_num"], step)

    # Final summary
    if completed_steps == total_steps and total_steps > 0:
        st.markdown("---")
        st.success(
            "🎉 **All steps completed!** You can review results above or re-run any step with modifications."
        )

        # Export all results
        if st.button("📦 Export All Results", key="export_all"):
            st.info("Export functionality coming soon!")


def render_sequential_interactive_mode(analysis_steps):
    """Render sequential interactive mode with chat-like interface."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "completed_processes" not in st.session_state:
        st.session_state.completed_processes = {}
    if "is_streaming" not in st.session_state:
        st.session_state.is_streaming = False
    if "should_run_agent" not in st.session_state:
        st.session_state.should_run_agent = False

    st.markdown("### 💬 Analysis Conversation")

    if st.button(
        "Start Analysis",
        key="start_analysis_btn",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.analysis_started = True
        st.session_state.should_run_agent = True
        st.rerun()

    if not st.session_state.get("analysis_started", False):
        return

    if not st.session_state.chat_history:
        with st.chat_message(
            "assistant", avatar=f"{CURRENT_ABS_DIR}/logo/AI_assistant_logo.png"
        ):
            st.markdown("👋 **Hi! I am OmicsHorizon, your bioinformatics assistant.**")

    for message in st.session_state.chat_history:
        with st.chat_message(
            message["role"],
            avatar=(
                f"{CURRENT_ABS_DIR}/logo/AI_assistant_logo.png"
                if message["role"] == "assistant"
                else None
            ),
        ):
            if message["role"] == "assistant":
                st.markdown(format_agent_output_for_display(message["content"]))
            else:
                st.markdown(message["content"])
            if message.get("files"):
                display_chat_files(message["files"])

    user_input = st.chat_input("메시지를 입력하세요...", key="user_chat_input")
    if user_input:
        add_chat_message("user", user_input)
        st.session_state.should_run_agent = True
        st.rerun()

    os.chdir(st.session_state.work_dir)
    data_info = ", ".join([f"`{f}`" for f in st.session_state.data_files])
    prompt = f"""Perform bioinformatics analysis.
#Analysis Instructions:
{st.session_state.analysis_method}

DATA FILES: {data_info}

DATA BRIEFING:
{st.session_state.data_briefing if st.session_state.data_briefing else "Files are available in the working directory"}

"""

    has_assistant_history = any(
        msg.get("role") == "assistant" for msg in st.session_state.chat_history
    )
    agent_input = build_agent_input_from_history(
        initial_prompt=prompt, include_initial=not has_assistant_history
    )

    if not st.session_state.should_run_agent:
        return

    st.session_state.is_streaming = True
    attachments: list[str] = []
    result_text = ""
    with st.chat_message(
        "assistant", avatar=f"{CURRENT_ABS_DIR}/logo/AI_assistant_logo.png"
    ):
        message_placeholder = st.empty()
        baseline_files = _collect_workspace_artifacts(CHAT_ATTACHMENT_PATTERNS)
        with st.spinner("AI is performing the analysis..."):
            try:
                message_stream = st.session_state.agent.go_stream(agent_input)
                for chunk in message_stream:
                    node = chunk[1][1]["langgraph_node"]
                    chunk_data = chunk[1][0]
                    if node not in {"generate", "execute"} or not hasattr(
                        chunk_data, "content"
                    ):
                        continue
                    content = chunk_data.content
                    if isinstance(content, list):
                        content = "".join(
                            item for item in content if isinstance(item, str)
                        )
                    if not content:
                        continue
                    result_text += content
                    message_placeholder.markdown(
                        format_agent_output_for_display(result_text)
                    )
            except Exception as exc:  # pragma: no cover - UI fallback
                st.error(f"Agent execution failed: {exc}")
            finally:
                st.session_state.is_streaming = False

        updated_files = _collect_workspace_artifacts(CHAT_ATTACHMENT_PATTERNS)
        new_files = sorted(
            updated_files - baseline_files, key=lambda path: os.path.getmtime(path)
        )
        attachments = new_files
        if result_text:
            message_placeholder.markdown(
                format_agent_output_for_display(result_text)
            )
        if new_files:
            display_chat_files(new_files)

    st.session_state.should_run_agent = False
    if result_text.strip():
        maybe_add_assistant_message(result_text, files=attachments)
    st.rerun()


def render_workflow_start_screen(analysis_steps):
    """Render the workflow start screen"""
    st.markdown(f"## {t('ready_to_start')}")

    total_steps = len(analysis_steps)

    st.markdown(f"**{t('total_steps')}** {total_steps}")
    st.markdown(f"**{t('workflow_overview')}**")

    for i, step in enumerate(analysis_steps, 1):
        status = st.session_state.steps_state.get(step["step_num"], {}).get(
            "status", "pending"
        )
        status_icon = {
            "completed": "✅",
            "in_progress": "⚙️",
            "pending": "⏳",
            "error": "❌",
        }.get(status, "⏳")
        st.markdown(f"{i}. {status_icon} {step['title']}")

    st.markdown("---")

    if st.button(
        t("start_analysis"),
        key="start_workflow",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.current_step = 1
        st.rerun()


def render_current_step_sequential(analysis_steps):
    """Render the current step in sequential mode"""
    current_step_num = st.session_state.current_step
    current_step_data = next(
        (s for s in analysis_steps if s["step_num"] == current_step_num), None
    )

    if not current_step_data:
        st.error(f"Step {current_step_num} not found.")
        return

    # Step header
    step_status = st.session_state.steps_state.get(current_step_num, {}).get(
        "status", "pending"
    )

    if step_status == "completed":
        # Step completed - show results and feedback options
        render_step_completion_interface(current_step_data)
    else:
        # Step not completed - show step execution interface
        render_step_execution_interface(current_step_data)


def render_step_execution_interface(step_data):
    """Render interface for executing a step"""
    step_num = step_data["step_num"]
    step_title = step_data["title"]

    # Initialize step state if not exists
    if step_num not in st.session_state.steps_state:
        st.session_state.steps_state[step_num] = {
            "status": "pending",
            "title": step_title,
            "description": step_data.get("description", ""),
            "result": None,
            "solution": None,
            "formatted_process": None,
            "files": [],
            "feedback": None,
            "iteration": 0,
        }

    st.markdown(t("step_execution").format(step_num=step_num, step_title=step_title))

    # Step description
    if step_data.get("description"):
        st.info(f"**Description:** {step_data['description']}")

    # Previous steps summary
    if step_num > 1:
        st.markdown(f"### {t('previous_steps_summary')}")
        prev_steps = []
        for i in range(1, step_num):
            step_info = st.session_state.steps_state.get(i, {})
            if step_info.get("status") == "completed":
                prev_steps.append(f"✅ Step {i}: {step_info.get('title', 'Unknown')}")

        if prev_steps:
            for step_summary in prev_steps[-3:]:  # Show last 3 steps
                st.markdown(f"- {step_summary}")
        else:
            st.markdown("*No previous steps completed*")

        # Show available result files
        available_files = get_available_result_files(step_num)
        if available_files:
            st.markdown("### 📁 Available Result Files")
            for file_info in available_files[:5]:  # Show top 5
                step_indicator = (
                    f" (Step {file_info['associated_step']})"
                    if file_info["associated_step"]
                    else ""
                )
                st.markdown(f"• {file_info['name']}{step_indicator}")
            if len(available_files) > 5:
                st.markdown(f"*... and {len(available_files) - 5} more files*")
            st.markdown("*These files will be automatically available to this step.*")

        st.markdown("---")

    # Current step status and execution
    step_info = st.session_state.steps_state.get(step_num, {})

    if step_info.get("status") == "in_progress":
        # Step is marked as in progress - execute immediately
        st.info("⚙️ Step 실행 중...")
        with st.spinner("분석을 수행하고 있습니다..."):
            success = execute_single_step(step_num, step_data)
            if success:
                st.success(f"✅ Step {step_num} 완료!")
                st.rerun()
            else:
                st.error(f"❌ Step {step_num} 실행 실패")
                # Reset status to allow retry
                st.session_state.steps_state[step_num]["status"] = "pending"
    else:
        # Step not started - show execute button
        if st.button(
            t("execute_step").format(step_num=step_num),
            key=f"execute_step_{step_num}",
            type="primary",
            use_container_width=True,
        ):
            # Mark as in progress and rerun to trigger execution
            st.session_state.steps_state[step_num]["status"] = "in_progress"
            st.rerun()


def render_step_completion_interface(step_data):
    """Render interface for completed step with feedback options"""
    step_num = step_data["step_num"]
    step_title = step_data["title"]

    st.markdown(t("step_completed").format(step_num=step_num, step_title=step_title))

    # Show step results
    step_info = st.session_state.steps_state.get(step_num, {})
    if step_info.get("solution"):
        st.markdown("### 📊 Results")
        st.markdown(step_info["solution"])

    # Show generated files with download buttons
    if step_info.get("files"):
        st.markdown("### 📈 Generated Files")
        for file_path in step_info["files"]:
            filename = os.path.basename(file_path)
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                # Show image if it's an image file, otherwise show filename
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                    st.image(file_path, use_container_width=True, caption=filename)
                else:
                    st.markdown(f"📄 **{filename}**")

            with col2:
                # Show file size
                try:
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    st.caption(f"Size: {file_size:.1f} KB")
                except:
                    st.caption("Size: Unknown")

            with col3:
                # Download button
                try:
                    with open(file_path, "rb") as f:
                        file_data = f.read()
                    st.download_button(
                        label="⬇️ Download",
                        data=file_data,
                        file_name=filename,
                        mime="application/octet-stream",
                        key=f"download_current_{step_num}_{filename}",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Could not load {filename}")

    st.markdown("---")

    # Feedback interface
    st.markdown(f"### {t('step_feedback')}")

    # Check if this step already has feedback
    existing_feedback = st.session_state.step_feedback.get(step_num, "")

    feedback = st.text_area(
        t("step_feedback_placeholder").format(step_num=step_num),
        value=existing_feedback,
        key=f"feedback_step_{step_num}",
        placeholder=t("step_feedback_example"),
        height=100,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            t("modify_step"), key=f"modify_step_{step_num}", use_container_width=True
        ):
            if feedback and feedback.strip():
                st.session_state.step_feedback[step_num] = feedback.strip()
                # Re-execute step with feedback
                execute_single_step_with_feedback(step_num, step_data, feedback.strip())
                st.rerun()
            else:
                st.warning("Please provide feedback for modification.")

    with col2:
        if st.button(
            t("continue_to_next"),
            key=f"continue_step_{step_num}",
            type="primary",
            use_container_width=True,
        ):
            # Save feedback if provided
            if feedback and feedback.strip():
                st.session_state.step_feedback[step_num] = feedback.strip()

            # Move to next step
            next_step = step_num + 1
            total_steps = len(parse_analysis_steps(st.session_state.analysis_method))

            if next_step <= total_steps:
                st.session_state.current_step = next_step
            else:
                st.session_state.current_step = total_steps + 1  # Mark as completed

            st.rerun()

    with col3:
        if step_num > 1:
            if st.button(
                t("back_to_previous"),
                key=f"back_step_{step_num}",
                use_container_width=True,
            ):
                st.session_state.current_step = step_num - 1
                st.rerun()


def render_workflow_completion_screen(analysis_steps):
    """Render the workflow completion screen"""
    st.markdown(f"## {t('workflow_completed')}")

    total_steps = len(analysis_steps)
    completed_steps = sum(
        1
        for s in st.session_state.steps_state.values()
        if s.get("status") == "completed"
    )

    st.success(f"✅ All {total_steps} steps completed successfully!")

    # Summary of all steps
    st.markdown(f"### {t('workflow_summary')}")
    for step in analysis_steps:
        step_num = step["step_num"]
        step_info = st.session_state.steps_state.get(step_num, {})
        status = step_info.get("status", "pending")
        status_icon = {
            "completed": "✅",
            "in_progress": "⚙️",
            "pending": "⏳",
            "error": "❌",
        }.get(status, "⏳")

        feedback = st.session_state.step_feedback.get(step_num, "")
        feedback_indicator = " 💬" if feedback else ""

        st.markdown(
            f"{status_icon} Step {step_num}: {step['title']}{feedback_indicator}"
        )

    st.markdown("---")

    # Options for completed workflow
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            t("restart_workflow"), key="restart_workflow", use_container_width=True
        ):
            # Reset workflow
            st.session_state.steps_state = {}
            st.session_state.current_step = 0
            st.session_state.step_feedback = {}
            st.success("🔄 Workflow restarted!")
            st.rerun()

    with col2:
        if st.button(
            t("export_results"), key="export_results", use_container_width=True
        ):
            st.info("Export functionality coming soon!")

    with col3:
        if st.button(t("review_steps"), key="review_steps", use_container_width=True):
            # Go back to last step for review
            st.session_state.current_step = total_steps
            st.rerun()


def handle_chat_input(user_input, analysis_steps):
    """Handle user input in chat interface"""
    # Simple keyword-based response system
    user_input_lower = user_input.lower()

    if any(word in user_input_lower for word in ["다음", "next", "계속", "proceed"]):
        # Move to next step
        current_step = st.session_state.current_step
        total_steps = len(analysis_steps)

        if current_step < total_steps:
            next_step = current_step + 1
            st.session_state.current_step = next_step
            # Initialize next step state
            next_step_data = next(
                (s for s in analysis_steps if s["step_num"] == next_step), None
            )
            if next_step_data:
                initialize_step_state(next_step, next_step_data)
            add_chat_message(
                "assistant", f"✅ 다음 단계(Step {next_step})로 진행합니다!"
            )
        else:
            add_chat_message("assistant", "🎉 모든 단계가 완료되었습니다!")

    elif any(word in user_input_lower for word in ["이전", "previous", "back"]):
        # Go back to previous step
        current_step = st.session_state.current_step
        if current_step > 1:
            prev_step = current_step - 1
            st.session_state.current_step = prev_step
            add_chat_message(
                "assistant", f"⬅️ 이전 단계(Step {prev_step})로 돌아갑니다."
            )
        else:
            add_chat_message(
                "assistant", "❌ 첫 번째 단계입니다. 더 이전으로 갈 수 없습니다."
            )

    elif any(word in user_input_lower for word in ["다시", "rerun", "재실행", "retry"]):
        # Rerun current step
        add_chat_message(
            "assistant",
            f"🔄 현재 단계(Step {st.session_state.current_step})를 다시 실행합니다.",
        )
        # Reset current step status to allow rerun
        current_step = st.session_state.current_step
        if current_step in st.session_state.steps_state:
            st.session_state.steps_state[current_step]["status"] = "pending"
            st.session_state.steps_state[current_step]["iteration"] = 0
        # Ensure step state exists
        current_step_data = next(
            (s for s in analysis_steps if s["step_num"] == current_step), None
        )
        if current_step_data:
            initialize_step_state(current_step, current_step_data)

    elif any(word in user_input_lower for word in ["완료", "done", "finish", "종료"]):
        # Mark workflow as completed
        total_steps = len(analysis_steps)
        completed_steps = sum(
            1
            for s in st.session_state.steps_state.values()
            if s.get("status") == "completed"
        )

        if completed_steps == total_steps:
            add_chat_message(
                "assistant", "🎉 분석이 완료되었습니다! 결과를 확인해주세요."
            )
            st.session_state.current_step = total_steps + 1  # Mark as fully completed
        else:
            add_chat_message(
                "assistant",
                f"⚠️ 아직 {total_steps - completed_steps}개의 단계가 남아있습니다.",
            )

    else:
        # General feedback or question - pass to LLM for refinement
        add_chat_message("assistant", "💭 피드백을 처리하고 있습니다...")

        # Here we could call an LLM to process the feedback
        # For now, just acknowledge
        add_chat_message(
            "assistant",
            f"피드백 감사합니다: '{user_input}'\n\n현재 단계에서 이를 고려하여 분석을 진행하겠습니다.",
        )


def execute_single_step_with_feedback(step_num, step_info, feedback):
    """Execute a single step with additional feedback context"""

    # Initialize step state if not exists
    if step_num not in st.session_state.steps_state:
        st.session_state.steps_state[step_num] = {
            "status": "pending",
            "title": step_info["title"],
            "description": step_info.get("description", ""),
            "result": None,
            "solution": None,
            "formatted_process": None,
            "files": [],
            "feedback": None,
            "iteration": 0,
        }

    # Update step status
    st.session_state.steps_state[step_num]["status"] = "in_progress"
    st.session_state.steps_state[step_num]["iteration"] += 1

    # Get previous context
    previous_context = get_previous_context(step_num)

    # Get available result files from previous steps
    available_result_files = get_available_result_files(step_num)

    # Build prompt with feedback
    data_info = ", ".join([f"`{f}`" for f in st.session_state.data_files])

    # Add available result files to prompt
    result_files_info = ""
    if available_result_files:
        result_files_info = "\n\nAVAILABLE RESULT FILES FROM PREVIOUS STEPS:"
        for file_info in available_result_files[:10]:  # Limit to 10 most recent
            result_files_info += f"\n- {file_info['name']} ({file_info['extension']}, {file_info['size']/1024:.1f} KB)"
            if file_info["associated_step"]:
                result_files_info += f" - from Step {file_info['associated_step']}"

    prompt = f"""Perform Step {step_num} of the bioinformatics analysis with the following feedback applied.

DATA FILES: {data_info}

DATA BRIEFING:
{st.session_state.data_briefing if st.session_state.data_briefing else "Files are available in the working directory"}

{"PREVIOUS STEPS CONTEXT:" if previous_context else ""}
{previous_context}

{result_files_info}

CURRENT STEP {step_num}: {step_info['title']}
Description: {step_info.get('description', step_info.get('full_text', ''))}

🔄 USER FEEDBACK (apply these modifications):
{feedback}

INSTRUCTIONS:
- Apply the user's feedback to modify this step
- **IMPORTANT**: Check and utilize result files from previous steps when relevant
- Load any CSV/TSV/TXT files from previous steps if they contain processed data you need
- Reference previous analysis results to build upon existing work
- Save any plots with descriptive filenames (e.g., "step{step_num}_*.png")
- Provide detailed results in <solution> tag
- Include specific numbers and statistics

Execute the modified Step {step_num} now."""

    try:
        # Execute with agent
        result = process_with_agent(prompt, show_process=False, use_history=False)

        # Extract solution content
        solution_match = re.search(r"<solution>(.*?)</solution>", result, re.DOTALL)
        if solution_match:
            solution_content = solution_match.group(1).strip()

            # Clean up solution content
            solution_content = re.sub(
                r"<execute>.*?</execute>", "", solution_content, flags=re.DOTALL
            )
            solution_content = re.sub(
                r"<observation>.*?</observation>", "", solution_content, flags=re.DOTALL
            )
            solution_content = re.sub(
                r"<think>.*?</think>", "", solution_content, flags=re.DOTALL
            )
            solution_content = re.sub(
                r"```[a-z]*\n.*?```", "", solution_content, flags=re.DOTALL
            )
            solution_content = re.sub(
                r"^\s*\d+\.\s*\[[\s✓✗✅❌⬜]\].*?$",
                "",
                solution_content,
                flags=re.MULTILINE,
            )
            solution_content = re.sub(r"===.*?===", "", solution_content)
            solution_content = re.sub(
                r"🐍\s*\*\*코드 실행.*?\*\*", "", solution_content
            )
            solution_content = re.sub(
                r"✅\s*\*\*실행 성공.*?\*\*", "", solution_content
            )
            solution_content = re.sub(
                r"^---+$", "", solution_content, flags=re.MULTILINE
            )
            solution_content = re.sub(r"\n{3,}", "\n\n", solution_content).strip()
        else:
            observations = re.findall(
                r"<observation>(.*?)</observation>", result, re.DOTALL
            )
            solution_content = (
                observations[-1].strip()
                if observations
                else "Analysis completed. See process details below."
            )

        # Update step state
        st.session_state.steps_state[step_num]["status"] = "completed"
        st.session_state.steps_state[step_num]["result"] = result
        st.session_state.steps_state[step_num]["solution"] = solution_content
        st.session_state.steps_state[step_num]["formatted_process"] = (
            format_agent_output_for_display(result)
        )

        # Extract new files
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"]
        all_images = []
        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(st.session_state.work_dir, ext)))

        new_files = [f for f in all_images if f not in get_all_previous_files(step_num)]
        if new_files:
            st.session_state.steps_state[step_num]["files"].extend(new_files)

        # Don't add chat message here - it will be handled by render_sequential_interactive_mode

        return f"✅ Step {step_num} has been refined successfully!"

    except Exception as e:
        st.session_state.steps_state[step_num]["status"] = "error"
        st.session_state.steps_state[step_num]["result"] = f"Error: {str(e)}"
        return f"❌ Error executing refinement: {str(e)}"


def is_workflow_completed():
    """Check if the entire workflow has been completed"""
    if not st.session_state.steps_state or not st.session_state.analysis_method:
        return False

    # Parse analysis steps
    try:
        analysis_steps = parse_analysis_steps(st.session_state.analysis_method)
        total_steps = len(analysis_steps)
        completed_steps = sum(
            1
            for s in st.session_state.steps_state.values()
            if s.get("status") == "completed"
        )
        return completed_steps == total_steps and total_steps > 0
    except:
        return False


def apply_analysis_refinement(refinement_request):
    """Apply user refinement request to the completed analysis"""

    if not is_workflow_completed():
        return "❌ Analysis workflow is not yet completed. Please complete all steps first."

    # Get context from all completed steps
    context_parts = []

    # Add data briefing
    if st.session_state.data_briefing:
        context_parts.append("=== DATA BRIEFING ===")
        context_parts.append(st.session_state.data_briefing[:2000])  # Limit size

    # Add analysis method
    if st.session_state.analysis_method:
        context_parts.append("=== ANALYSIS WORKFLOW ===")
        context_parts.append(st.session_state.analysis_method[:1000])

    # Add completed steps results
    context_parts.append("=== COMPLETED ANALYSIS RESULTS ===")
    for step_num in sorted(st.session_state.steps_state.keys()):
        step_data = st.session_state.steps_state[step_num]
        if step_data["status"] == "completed":
            context_parts.append(f"--- Step {step_num}: {step_data['title']} ---")
            if step_data.get("solution"):
                # Truncate long results
                solution = step_data["solution"][:1500]
                if len(step_data["solution"]) > 1500:
                    solution += "... (truncated)"
                context_parts.append(f"Results: {solution}")

            # Mention generated files
            if step_data.get("files"):
                file_names = [os.path.basename(f) for f in step_data["files"]]
                context_parts.append(f"Generated files: {', '.join(file_names)}")

    full_context = "\n".join(context_parts)

    # Build refinement prompt
    prompt = f"""You are a bioinformatics analysis assistant. The user has completed a full analysis workflow and now wants to make refinements or modifications.

COMPLETED ANALYSIS CONTEXT:
{full_context[:4000]}  # Limit context size

USER REFINEMENT REQUEST:
{refinement_request}

INSTRUCTIONS:
- Understand what specific modification or refinement the user is requesting
- Apply the requested changes to the existing analysis results
- If the request involves re-running analysis with different parameters, suggest the appropriate modifications
- If the request is about figure modifications (labels, colors, etc.), provide guidance on how to implement them
- If the request requires code changes, provide the specific code modifications needed
- Be specific and actionable in your response
- Reference the existing results and suggest concrete next steps

Provide a detailed response on how to implement the requested refinement:"""

    try:
        llm = st.session_state.agent.llm
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        return f"Error processing refinement request: {str(e)}\n\nPlease try again."


def execute_refinement_action(refinement_request, target_step=None):
    """Execute a specific refinement action on a step"""

    if not target_step or target_step not in st.session_state.steps_state:
        return "❌ Invalid target step specified."

    step_data = st.session_state.steps_state[target_step]
    if step_data["status"] != "completed":
        return f"❌ Step {target_step} is not completed yet."

    # Build context for the specific step
    context = f"""
Step {target_step}: {step_data['title']}
Description: {step_data.get('description', '')}

Previous Results:
{step_data.get('solution', 'No previous results')}

Refinement Request: {refinement_request}
"""

    # Create refinement prompt
    data_info = ", ".join([f"`{f}`" for f in st.session_state.data_files])

    prompt = f"""Perform a refinement of Step {target_step} based on the user's request.

DATA FILES: {data_info}

PREVIOUS STEP RESULTS:
{context}

USER REFINEMENT REQUEST:
{refinement_request}

INSTRUCTIONS:
- Apply the requested modifications to Step {target_step}
- Use the existing results as a starting point
- Generate updated results with the requested changes
- Save any new plots with descriptive filenames including 'refined' suffix
- Provide detailed results in <solution> tag

Execute the refined Step {target_step} now."""

    try:
        # Execute refinement
        result = process_with_agent(prompt, show_process=False, use_history=False)

        # Update step data with refinement
        step_data["solution"] = extract_solution_content(result)
        step_data["formatted_process"] = format_agent_output_for_display(result)

        # Extract new files if any
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"]
        all_images = []
        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(st.session_state.work_dir, ext)))

        new_files = [
            f for f in all_images if f not in get_all_previous_files(target_step)
        ]
        if new_files:
            step_data["files"].extend(new_files)

        return f"✅ Step {target_step} has been refined successfully!"

    except Exception as e:
        return f"❌ Error executing refinement: {str(e)}"


def answer_qa_question(question):
    """Answer a Q&A question based on current analysis context"""

    # Get context
    context = get_qa_context()

    # Build prompt
    prompt = f"""You are a helpful bioinformatics analysis assistant. A user is asking a question about their ongoing analysis.

ANALYSIS CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Provide a clear, concise answer based on the analysis context
- If the information is not available in the context, say so politely
- Reference specific steps or results when relevant
- Be technical but understandable
- If the user asks for clarification about a method or result, explain it clearly
- If the user asks "why", provide reasoning based on the analysis

Answer the question:"""

    try:
        # Use agent's LLM to answer (without history, single-turn Q&A)
        llm = st.session_state.agent.llm
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        return (
            f"Error generating answer: {str(e)}\n\nPlease try rephrasing your question."
        )


def render_step_panel(step_num, step_info):
    """Render interactive panel for a single step"""

    # Initialize state if needed
    initialize_step_state(step_num, step_info)

    step_data = st.session_state.steps_state[step_num]
    status = step_data["status"]

    # Status emoji
    status_config = {
        "completed": {"emoji": "✅", "color": "#28a745"},
        "in_progress": {"emoji": "⚙️", "color": "#ffc107"},
        "pending": {"emoji": "⏸️", "color": "#6c757d"},
        "error": {"emoji": "❌", "color": "#dc3545"},
    }

    config = status_config[status]

    # Step header
    with st.expander(
        f"{config['emoji']} **Step {step_num}: {step_info['title']}**",
        expanded=(status in ["completed", "in_progress"]),
    ):
        # Method description
        if step_info.get("description"):
            with st.expander("📖 Method Description", expanded=False):
                st.info(step_info["description"])
            st.markdown("---")

        # Status badge
        st.markdown(f"**Status:** {config['emoji']} `{status.upper()}`")

        if step_data["iteration"] > 1:
            st.caption(f"🔄 Iteration: {step_data['iteration']}")

        st.markdown("---")

        # Display based on status
        if status == "completed":
            render_completed_step(step_num, step_data, step_info)

        elif status == "in_progress":
            st.info("⚙️ Step is currently executing... Please wait.")

        elif status == "pending":
            render_pending_step(step_num, step_info)

        elif status == "error":
            st.error(f"❌ Error occurred: {step_data['result']}")
            if st.button(f"🔄 Retry Step {step_num}", key=f"retry_{step_num}"):
                execute_single_step(step_num, step_info)
                st.rerun()


def render_completed_step(step_num, step_data, step_info):
    """Render completed step with results and controls"""

    # Results (Solution only - clean output)
    st.markdown("### 📊 Results")

    solution_content = step_data.get("solution", "")

    # FALLBACK: If solution field doesn't exist (old sessions), extract it now
    if not solution_content and step_data.get("result"):
        raw_result = step_data["result"]
        solution_match = re.search(r"<solution>(.*?)</solution>", raw_result, re.DOTALL)
        if solution_match:
            solution_content = solution_match.group(1).strip()
            # Apply same cleaning as in execute_single_step
            solution_content = re.sub(
                r"<execute>.*?</execute>", "", solution_content, flags=re.DOTALL
            )
            solution_content = re.sub(
                r"<observation>.*?</observation>", "", solution_content, flags=re.DOTALL
            )
            solution_content = re.sub(
                r"<think>.*?</think>", "", solution_content, flags=re.DOTALL
            )
            solution_content = re.sub(
                r"```[a-z]*\n.*?```", "", solution_content, flags=re.DOTALL
            )
            solution_content = re.sub(
                r"^\s*\d+\.\s*\[[\s✓✗✅❌⬜]\].*?$",
                "",
                solution_content,
                flags=re.MULTILINE,
            )
            solution_content = re.sub(r"===.*?===", "", solution_content)
            solution_content = re.sub(
                r"🐍\s*\*\*코드 실행.*?\*\*", "", solution_content
            )
            solution_content = re.sub(
                r"✅\s*\*\*실행 성공.*?\*\*", "", solution_content
            )
            solution_content = re.sub(
                r"^---+$", "", solution_content, flags=re.MULTILINE
            )
            solution_content = re.sub(r"\n{3,}", "\n\n", solution_content).strip()
            # Save it for next time
            step_data["solution"] = solution_content

    if solution_content:
        # Display solution content
        st.markdown(solution_content)

        # Check if solution references any images and display them inline
        if step_data["files"]:
            mentioned_images = []
            for img_path in step_data["files"]:
                img_name = os.path.basename(img_path)
                # Check if image is mentioned in solution
                if (
                    img_name in solution_content
                    or img_name.replace("_", " ") in solution_content.lower()
                ):
                    mentioned_images.append(img_path)

            # Display mentioned images inline with results
            if mentioned_images:
                for img_path in mentioned_images:
                    st.image(
                        img_path,
                        use_container_width=True,
                        caption=os.path.basename(img_path),
                    )
    else:
        st.info("No results available. See analysis process below.")

    st.markdown("---")

    # Analysis Process (Full details in expander)
    with st.expander("🔍 View Analysis Process", expanded=False):
        formatted_process = step_data.get("formatted_process", "")
        if formatted_process:
            # Remove the solution section from formatted process to avoid duplication
            # since solution is already displayed above
            formatted_process_clean = re.sub(
                r"---\s*\n\s*🎯 \*\*최종 답변:\*\*\s*\n.*?(?=---|\Z)",
                "",
                formatted_process,
                flags=re.DOTALL,
            ).strip()

            if formatted_process_clean:
                st.markdown(formatted_process_clean)
            else:
                st.info("Process details contain only final results (shown above)")
        else:
            st.info("Process details not available")

    st.markdown("---")

    # Figures (All generated figures)
    if step_data["files"]:
        st.markdown(f"### 📈 Figures ({len(step_data['files'])})")

        cols = st.columns(min(3, len(step_data["files"])))
        for idx, img_path in enumerate(step_data["files"]):
            with cols[idx % 3]:
                st.image(
                    img_path,
                    use_container_width=True,
                    caption=os.path.basename(img_path),
                )

                # Download button
                with open(img_path, "rb") as f:
                    file_data = f.read()
                st.download_button(
                    label="⬇️ Download",
                    data=file_data,
                    file_name=os.path.basename(img_path),
                    mime="image/png",
                    key=f"download_step_{step_num}_{idx}",
                )

    st.markdown("---")

    # Feedback and controls
    st.markdown("### 🎮 Controls")

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        feedback = st.text_input(
            "💬 Feedback or modifications:",
            key=f"feedback_input_{step_num}",
            placeholder="e.g., 'Use stricter p-value threshold' or 'Add sample labels to plot'",
            help="Provide natural language feedback to modify this step",
        )

    with col2:
        if st.button("🔄 Re-run", key=f"rerun_{step_num}", use_container_width=True):
            if feedback:
                st.session_state.steps_state[step_num]["feedback"] = feedback
            execute_single_step(step_num, step_info)
            st.rerun()

    with col3:
        # Check if this is the last step
        total_steps = len(parse_analysis_steps(st.session_state.analysis_method))

        if step_num < total_steps:
            if st.button(
                f"▶️ Next",
                key=f"next_{step_num}",
                use_container_width=True,
                type="primary",
            ):
                # Initialize next step as pending (unlocked)
                st.rerun()
        else:
            st.success("🎉 All steps completed!")


def render_pending_step(step_num, step_info):
    """Render pending step with start button"""

    # Check if previous step is completed
    can_start = False

    if step_num == 1:
        can_start = True
    else:
        prev_step = st.session_state.steps_state.get(step_num - 1)
        if prev_step and prev_step["status"] == "completed":
            can_start = True

    if can_start:
        st.info(f"⏸️ Ready to execute Step {step_num}")

        if st.button(
            f"▶️ Start Step {step_num}",
            key=f"start_{step_num}",
            type="primary",
            use_container_width=True,
        ):
            execute_single_step(step_num, step_info)
            st.rerun()
    else:
        st.warning(f"🔒 Waiting for Step {step_num - 1} to complete")


# =============================================================================
# INTELLIGENT PARSING FUNCTIONS (Agent-Trusting Approach)
# =============================================================================


def fuzzy_match_steps(agent_plan_items, expected_steps, threshold=0.6):
    """
    Agent가 만든 plan을 expected_steps와 fuzzy matching

    Args:
        agent_plan_items: Agent가 생성한 plan 항목들 (checkbox에서 추출)
        expected_steps: 사용자가 정의한 분석 단계들
        threshold: 매칭 임계값 (0-1)

    Returns:
        dict: {expected_step_num: agent_plan_num}
    """
    from difflib import SequenceMatcher

    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    mapping = {}
    used_agent_items = set()

    for exp_step in expected_steps:
        exp_num = exp_step["step_num"]
        exp_title = exp_step["title"].lower()

        best_match = None
        best_score = 0

        for agent_num, agent_item in agent_plan_items.items():
            if agent_num in used_agent_items:
                continue

            agent_title = agent_item["title"].lower()
            score = similarity(exp_title, agent_title)

            # Also check keywords
            exp_keywords = set(exp_title.split())
            agent_keywords = set(agent_title.split())
            keyword_overlap = len(exp_keywords & agent_keywords) / max(
                len(exp_keywords), 1
            )

            # Combined score
            final_score = 0.7 * score + 0.3 * keyword_overlap

            if final_score > best_score and final_score >= threshold:
                best_score = final_score
                best_match = agent_num

        if best_match:
            mapping[exp_num] = best_match
            used_agent_items.add(best_match)

    return mapping


def extract_plan_from_output(agent_output):
    """
    Agent output에서 plan items 추출 (중복 제거)

    Returns:
        dict: {plan_num: {'title': str, 'status': str, 'position': int}}
    """
    # Checkbox 패턴 찾기 (기존 parse_step_progress와 동일)
    pattern = r"^\s*(\d+)\.\s*(?:\[([✓✗ ])\]|([✅❌⬜]))\s*(.+?)(?:\s*\(.*?\))?$"
    matches = re.finditer(pattern, agent_output, re.MULTILINE)

    plan_items = {}
    for match in matches:
        num = int(match.group(1))
        old_status = match.group(2)
        emoji_status = match.group(3)
        title = match.group(4).strip()
        position = match.start()

        # Determine status
        if old_status == "✓" or emoji_status == "✅":
            status = "completed"
        elif old_status == "✗" or emoji_status == "❌":
            status = "failed"
        else:
            status = "pending"

        # Update dict (later occurrences override earlier ones to handle plan updates)
        plan_items[num] = {
            "title": title,
            "status": status,
            "position": position,  # Note: position will be the LAST occurrence
        }

    return plan_items


def collect_step_content(agent_output, plan_items, step_mapping):
    """
    각 step의 내용 수집 (observations, executions)

    Args:
        agent_output: 전체 agent 출력
        plan_items: Agent의 plan items
        step_mapping: expected_step -> agent_plan 매핑

    Returns:
        dict: {expected_step_num: {'observations': [], 'executions': []}}
    """
    step_contents = {}

    for exp_num, agent_num in step_mapping.items():
        if agent_num not in plan_items:
            continue

        # 이 step의 시작 위치
        start_pos = plan_items[agent_num]["position"]

        # 다음 step의 시작 위치 (또는 끝)
        next_agent_nums = [n for n in plan_items.keys() if n > agent_num]
        if next_agent_nums:
            end_pos = plan_items[min(next_agent_nums)]["position"]
        else:
            end_pos = len(agent_output)

        # 이 구간에서 observations와 executions 추출
        section = agent_output[start_pos:end_pos]

        observations = re.findall(
            r"<observation>(.*?)</observation>", section, re.DOTALL
        )

        executions = re.findall(r"<execute>(.*?)</execute>", section, re.DOTALL)

        step_contents[exp_num] = {
            "observations": [obs.strip() for obs in observations],
            "executions": [exe.strip() for exe in executions],
            "agent_title": plan_items[agent_num]["title"],
            "status": plan_items[agent_num]["status"],
        }

    return step_contents


def assign_images_to_steps_smartly(step_contents, all_images, num_steps):
    """
    이미지를 step에 지능적으로 할당

    Strategy:
    1. 파일명에서 step 번호 감지 (step1_, step2_ 등)
    2. 그 외는 생성 시간 순서로 균등 분배
    """
    # 각 step에 이미지 리스트 초기화
    for step_num in step_contents:
        step_contents[step_num]["images"] = []

    unassigned_images = []

    # 1단계: 파일명 기반 할당
    for img_path in all_images:
        filename = os.path.basename(img_path).lower()
        assigned = False

        # step 번호 찾기
        step_match = re.search(r"step[_\s]?(\d+)", filename)
        if step_match:
            step_num = int(step_match.group(1))
            if step_num in step_contents:
                step_contents[step_num]["images"].append(img_path)
                assigned = True

        if not assigned:
            unassigned_images.append(img_path)

    # 2단계: 나머지 이미지를 시간 순서로 분배
    if unassigned_images:
        # 생성 시간 순 정렬
        unassigned_images.sort(key=lambda x: os.path.getctime(x))

        step_nums = sorted(step_contents.keys())
        if len(step_nums) > 0:
            images_per_step = len(unassigned_images) / len(step_nums)

            for idx, img in enumerate(unassigned_images):
                # 어느 step에 속하는지 계산
                step_idx = min(int(idx / images_per_step), len(step_nums) - 1)
                step_num = step_nums[step_idx]
                step_contents[step_num]["images"].append(img)

    return step_contents


def fallback_simple_distribution(agent_output, expected_steps, all_images):
    """
    Plan 파싱 실패 시 fallback: 단순 시간 기반 분배
    """
    step_contents = {}

    # 모든 observations 추출
    all_observations = re.findall(
        r"<observation>(.*?)</observation>", agent_output, re.DOTALL
    )

    # 균등 분배
    obs_per_step = (
        len(all_observations) / len(expected_steps) if len(expected_steps) > 0 else 0
    )
    img_per_step = (
        len(all_images) / len(expected_steps) if len(expected_steps) > 0 else 0
    )

    for idx, step in enumerate(expected_steps):
        step_num = step["step_num"]

        obs_start = int(idx * obs_per_step)
        obs_end = (
            int((idx + 1) * obs_per_step)
            if idx < len(expected_steps) - 1
            else len(all_observations)
        )

        img_start = int(idx * img_per_step)
        img_end = (
            int((idx + 1) * img_per_step)
            if idx < len(expected_steps) - 1
            else len(all_images)
        )

        observations = all_observations[obs_start:obs_end] if all_observations else []
        images = (
            sorted(all_images, key=lambda x: os.path.getctime(x))[img_start:img_end]
            if all_images
            else []
        )

        step_contents[step_num] = {
            "expected_title": step["title"],
            "agent_title": None,
            "observations": observations,
            "executions": [],
            "images": images,
            "status": "completed",
            "summary": "Content distributed based on temporal order.",
        }

    return step_contents


def parse_agent_output_intelligently(agent_output, expected_steps, all_images):
    """
    Agent 출력을 지능적으로 파싱 (Agent를 신뢰하는 접근법)

    Args:
        agent_output: 전체 agent 출력
        expected_steps: 사용자 정의 분석 단계
        all_images: 생성된 이미지 파일 목록

    Returns:
        dict: {step_num: {
            'expected_title': str,
            'agent_title': str,
            'observations': [str],
            'executions': [str],
            'images': [str],
            'status': str,
            'summary': str
        }}
    """
    # 1. Agent의 plan 추출
    plan_items = extract_plan_from_output(agent_output)

    if not plan_items:
        # Plan이 없으면 fallback: 시간 기반 단순 분배
        return fallback_simple_distribution(agent_output, expected_steps, all_images)

    # 2. Expected steps와 agent plan 매칭
    step_mapping = fuzzy_match_steps(plan_items, expected_steps)

    # 3. 각 step의 내용 수집
    step_contents = collect_step_content(agent_output, plan_items, step_mapping)

    # 4. 이미지 할당
    step_contents = assign_images_to_steps_smartly(
        step_contents, all_images, len(expected_steps)
    )

    # 5. 각 step에 expected_title 추가 및 summary 생성
    for exp_step in expected_steps:
        step_num = exp_step["step_num"]

        if step_num not in step_contents:
            # 매칭 실패한 step
            step_contents[step_num] = {
                "expected_title": exp_step["title"],
                "agent_title": None,
                "observations": [],
                "executions": [],
                "images": [],
                "status": "not_found",
                "summary": "This step was not clearly identified in the agent output.",
            }
        else:
            step_contents[step_num]["expected_title"] = exp_step["title"]

            # Summary 생성
            obs_count = len(step_contents[step_num]["observations"])
            img_count = len(step_contents[step_num]["images"])

            if obs_count > 0:
                # 첫 번째 observation의 일부를 summary로
                first_obs = step_contents[step_num]["observations"][0]
                summary = first_obs[:200] + "..." if len(first_obs) > 200 else first_obs
            else:
                summary = f"Completed with {img_count} figure(s) generated."

            step_contents[step_num]["summary"] = summary

    return step_contents


def display_structured_analysis_results(
    result_text, analysis_steps, title="Analysis Results"
):
    """Display results with intelligent agent-trusting approach.

    Args:
        result_text: Raw agent output
        analysis_steps: List of steps from parse_analysis_steps()
        title: Display title
    """
    st.markdown(f"### {title}")

    if not result_text or len(result_text) < MIN_MEANINGFUL_CONTENT_LENGTH:
        st.warning("No results to display")
        return

    # Get all generated images
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"]
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(st.session_state.work_dir, ext)))

    # Sort by creation time
    all_images.sort(key=os.path.getctime)

    # Use intelligent parsing (Agent-Trusting Approach)
    step_results = parse_agent_output_intelligently(
        result_text, analysis_steps, all_images
    )

    # Create tabs
    tab1, tab2, tab3 = st.tabs([t("step_by_step"), t("full_report"), t("raw_output")])

    with tab1:
        st.info(
            "💡 **Tip:** Results organized by agent's execution plan with intelligent matching"
        )
        st.markdown("---")

        # Display each step with intelligent results
        for idx, step_num in enumerate(sorted(step_results.keys())):
            result = step_results[step_num]
            expected_title = result["expected_title"]
            agent_title = result.get("agent_title")
            status = result["status"]
            observations = result["observations"]
            images = result["images"]
            summary = result["summary"]

            # Step header with expander
            with st.expander(
                f"**Step {step_num}: {expected_title}**", expanded=(idx < 3)
            ):
                # Show agent's interpretation if available
                if agent_title and agent_title.lower() != expected_title.lower():
                    st.caption(f"🤖 Agent identified as: _{agent_title}_")

                # Status badge
                status_emoji = {
                    "completed": "✅",
                    "failed": "❌",
                    "pending": "⏳",
                    "not_found": "❓",
                }
                st.markdown(f"{status_emoji.get(status, '❓')} **Status:** {status}")
                st.markdown("---")

                # Method description from Panel 2 (if available)
                step_obj = next(
                    (s for s in analysis_steps if s["step_num"] == step_num), None
                )
                if step_obj and step_obj.get("description"):
                    with st.expander("📝 Method Description", expanded=False):
                        st.info(step_obj["description"])

                # Summary
                st.markdown("##### 📝 Summary")
                st.info(summary)

                # Detailed observations
                if observations:
                    st.markdown(
                        f"##### 🔬 Detailed Results ({len(observations)} observations)"
                    )
                    for obs_idx, obs in enumerate(observations, 1):
                        with st.expander(
                            f"Observation {obs_idx}", expanded=(obs_idx == 1)
                        ):
                            st.markdown(obs)
                else:
                    st.markdown("##### 🔬 Detailed Results")
                    st.markdown("_No detailed observations captured for this step._")

                # Figures
                if images:
                    st.markdown(f"##### 📊 Figures ({len(images)})")
                    cols = st.columns(min(2, len(images)))
                    for img_idx, img_path in enumerate(images):
                        with cols[img_idx % 2]:
                            st.image(
                                img_path,
                                use_container_width=True,
                                caption=os.path.basename(img_path),
                            )

                            # Download button
                            with open(img_path, "rb") as f:
                                file_data = f.read()
                            st.download_button(
                                label="⬇️ Download",
                                data=file_data,
                                file_name=os.path.basename(img_path),
                                mime="image/png",
                                key=f"download_step_{step_num}_{img_idx}",
                            )
                else:
                    st.markdown("##### 📊 Figures")
                    st.info("_No figures generated for this step._")

                st.markdown("---")

    with tab2:
        st.markdown("##### Complete Analysis Report")
        st.info("📖 This shows the full analysis with all details, code, and results.")

        # Use formatted analysis process (already nicely formatted from format_agent_output_for_display)
        if (
            hasattr(st.session_state, "analysis_process")
            and st.session_state.analysis_process
        ):
            # Display the formatted process with collapsible code blocks
            with st.expander(
                "🔍 View Complete Analysis Process (with code)", expanded=True
            ):
                st.markdown(st.session_state.analysis_process)
        else:
            # Fallback: Extract and show solution content
            solution_match = re.search(
                r"<solution>(.*?)</solution>", result_text, flags=re.DOTALL
            )

            if solution_match:
                solution_content = solution_match.group(1).strip()

                # Minimal cleaning - just remove XML tags
                solution_content = re.sub(
                    r"<execute>.*?</execute>", "", solution_content, flags=re.DOTALL
                )
                solution_content = re.sub(
                    r"<observation>.*?</observation>",
                    "",
                    solution_content,
                    flags=re.DOTALL,
                )

                if solution_content and len(solution_content) > 50:
                    st.markdown(solution_content)
                else:
                    st.warning(
                        "⚠️ No structured results found. Please check the 'Raw Output' tab."
                    )
            else:
                st.warning(
                    "⚠️ No solution content found. Please check the 'Raw Output' tab."
                )

        # Show all images in a clean gallery
        if all_images:
            st.markdown("---")
            st.markdown("##### 📊 All Generated Figures")

            # Show images in a grid
            cols = st.columns(3)
            for idx, img_path in enumerate(all_images):
                with cols[idx % 3]:
                    st.image(
                        img_path,
                        use_container_width=True,
                        caption=os.path.basename(img_path),
                    )

                    # Download button - read file data outside of context manager
                    with open(img_path, "rb") as f:
                        file_data = f.read()
                    st.download_button(
                        label="⬇️ Download",
                        data=file_data,
                        file_name=os.path.basename(img_path),
                        mime="image/png",
                        key=f"download_full_{idx}",
                    )
        else:
            st.markdown("---")
            st.info("_No figures were generated during this analysis._")

    with tab3:
        # Raw output for debugging
        st.markdown("##### Raw Agent Output")
        st.info(
            "This shows the complete unprocessed output from the agent, useful for debugging."
        )

        # Show in expandable code block
        with st.expander("View Raw Output", expanded=False):
            max_raw_output = 10000
            st.code(
                (
                    result_text[:max_raw_output]
                    if len(result_text) > max_raw_output
                    else result_text
                ),
                language="text",
            )


def display_clean_result(result_text, title="Analysis Results"):
    """Display results in a beautiful, structured format.

    Legacy function - kept for compatibility.
    """
    st.markdown(f"### {title}")

    # Extract the clean content from solution/observation tags
    cleaned_text = extract_solution_content(result_text)

    if not cleaned_text:
        st.warning("No results to display")
        return

    # Additional aggressive cleaning for display
    # Remove any remaining prompt text
    prompt_markers = [
        r"Analyze these files.*?</solution>",
        r"Files:.*?\n",
        r"Output format:.*?\n",
        r"\[dimensions.*?\]",
        r"\[top 5-10.*?\]",
        r"Read this paper.*?numbered list\)",
    ]
    for marker in prompt_markers:
        cleaned_text = re.sub(marker, "", cleaned_text, flags=re.DOTALL | re.IGNORECASE)

    # Remove code-like patterns
    cleaned_text = re.sub(r"<execute>.*?</execute>", "", cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r"```[\s\S]*?```", "", cleaned_text)

    # Parse into sections
    sections = parse_structured_sections(cleaned_text)

    # Filter out empty or placeholder sections
    MIN_SECTION_CONTENT = 10
    sections = {
        k: v
        for k, v in sections.items()
        if v and len(v.strip()) > MIN_SECTION_CONTENT and not v.startswith("[")
    }

    if not sections:
        st.info("Analysis completed. Results are being processed...")
        return

    # Create tabs for different views
    tab1, tab2 = st.tabs(["📊 Summary", "📝 Full Report"])

    with tab1:
        # Display only the most important sections in summary
        priority_sections = [
            "Data Overview",
            "Overview",
            "Summary",
            "Analysis Summary",
            "Key Variables",
            "Results",
            "Key Results",
        ]

        for section_name in priority_sections:
            if section_name in sections:
                st.markdown(f"**{section_name}**")
                content = sections[section_name]

                # Display in clean format
                lines = [l.strip() for l in content.split("\n") if l.strip()]
                for line in lines[:5]:  # Show first 5 lines only in summary
                    if line.startswith(("-", "*", "•", "1.", "2.", "3.")):
                        st.markdown(f"  {line}")
                    else:
                        st.markdown(line)

                st.markdown("")

        # Show other sections as collapsed
        other_sections = [k for k in sections.keys() if k not in priority_sections]
        if other_sections:
            with st.expander("📂 More Details", expanded=False):
                for section_name in other_sections[:3]:
                    st.markdown(f"**{section_name}**")
                    st.markdown(sections[section_name])
                    st.markdown("---")

    with tab2:
        # Display all sections cleanly
        for section_name, content in sections.items():
            if section_name != "General":
                st.markdown(f"### {section_name}")
            st.markdown(content)
            st.markdown("")


def display_images_in_directory():
    """Display all images in the work directory."""
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"]
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(st.session_state.work_dir, ext)))

    if images:
        st.markdown("### 📊 Generated Visualizations")

        # Sort images by creation time (newest first)
        images.sort(key=os.path.getctime, reverse=True)

        cols = st.columns(min(3, len(images)))
        for idx, img_path in enumerate(images):
            with cols[idx % 3]:
                st.image(
                    img_path,
                    use_container_width=True,
                    caption=os.path.basename(img_path),
                )
                # Download button - read file data outside of context manager
                with open(img_path, "rb") as f:
                    file_data = f.read()
                st.download_button(
                    label=f"⬇️ Download",
                    data=file_data,
                    file_name=os.path.basename(img_path),
                    mime="image/png",
                    key=f"download_{os.path.basename(img_path)}",
                )




# =============================================================================
# INTERACTIVE MODE: STEP-BY-STEP EXECUTION
# =============================================================================


def initialize_step_state(step_num, step_info):
    """Initialize state for a single step"""
    if step_num not in st.session_state.steps_state:
        st.session_state.steps_state[step_num] = {
            "status": "pending",  # 'pending', 'in_progress', 'completed', 'error'
            "title": step_info["title"],
            "description": step_info.get("description", ""),
            "result": None,  # Raw agent output
            "solution": None,  # Extracted solution content (clean)
            "formatted_process": None,  # Fully formatted process for expander
            "files": [],
            "feedback": None,
            "iteration": 0,
        }


def get_previous_context(step_num):
    """Get context from previous steps to pass to current step"""
    if step_num == 1:
        return ""

    context_parts = []

    for i in range(1, step_num):
        if i in st.session_state.steps_state:
            step_data = st.session_state.steps_state[i]

            if step_data["status"] == "completed" and step_data["result"]:
                # Extract key information from previous step
                context_parts.append(f"=== Previous Step {i}: {step_data['title']} ===")

                # Extract observations
                observations = re.findall(
                    r"<observation>(.*?)</observation>", step_data["result"], re.DOTALL
                )

                if observations:
                    # Use last observation (usually the most relevant)
                    last_obs = observations[-1].strip()
                    # Truncate if too long
                    if len(last_obs) > 500:
                        last_obs = last_obs[:500] + "... (truncated)"
                    context_parts.append(f"Key findings: {last_obs}")

                # Include generated files
                if step_data["files"]:
                    file_list = ", ".join(
                        [os.path.basename(f) for f in step_data["files"]]
                    )
                    context_parts.append(f"Generated files: {file_list}")

                context_parts.append("")  # Empty line

    return "\n".join(context_parts)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # When run directly (not from LIMS)
    run_omicshorizon_app(from_lims=False, workspace_path=None)
