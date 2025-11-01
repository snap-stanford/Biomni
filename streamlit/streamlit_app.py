import gzip
import io
import logging
import os
import re
import shutil
import sys
import glob
import tempfile
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

_FINAL_ANSWER_SECTION_RE = re.compile(
    r"\n?\s*---\s*\n\s*🎯 \*\*최종 답변:\*\*\s*\n.*?(?:\n\s*---\s*)?",
    re.DOTALL,
)

_LOG_PREVIEW_LIMIT = 800


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


def _format_process_without_solution(raw_text: str) -> str:
    """Return formatted process output with the final answer section stripped."""
    formatted = format_agent_output_for_display(raw_text)
    cleaned = _FINAL_ANSWER_SECTION_RE.sub("\n\n", formatted)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _log_text_preview(context_label: str, text: Optional[str]) -> None:
    """Log a trimmed, single-line preview of text shown to the user."""
    if text is None:
        log.info("%s: <none>", context_label)
        return

    normalized = " ".join(text.split())
    if len(normalized) > _LOG_PREVIEW_LIMIT:
        normalized = f"{normalized[:_LOG_PREVIEW_LIMIT]}... (truncated)"
    log.info("%s: %s", context_label, normalized if normalized else "<empty>")

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

    # Launch conversational analysis once data and workflow are ready
    if st.session_state.data_files and st.session_state.analysis_method:
        render_analysis_conversation()
    elif not st.session_state.data_files:
        st.session_state.analysis_started = False
        st.session_state.should_run_agent = False
        st.warning("⚠️ Please upload data files in Panel 1")
    elif not st.session_state.analysis_method:
        st.session_state.analysis_started = False
        st.session_state.should_run_agent = False
        st.warning("⚠️ Please upload a paper or define analysis method in Panel 2")

    # Sidebar
    with st.sidebar:
        # Q&A Section at the very top
        st.markdown(f"### {t('qa_title')}")

        # Check if there's any analysis to ask about
        has_analysis = any(
            msg.get("role") == "assistant" for msg in st.session_state.chat_history
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
                        st.caption("• 현재까지의 주요 결과를 요약해줄 수 있나요?")
                        st.caption("• 사용된 통계 검정의 의미는 무엇인가요?")
                        st.caption("• volcano plot을 어떻게 해석해야 하나요?")
                        st.caption("• 이 p-value는 어떤 임계값과 비교해야 하나요?")
                    else:
                        st.caption("**Example questions:**")
                        st.caption("• Can you summarize the key findings so far?")
                        st.caption("• Why was this statistical test chosen?")
                        st.caption("• How should I interpret the volcano plot?")
                        st.caption("• What threshold should I compare this p-value against?")
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

        # Clear all button
        if st.button(t("clear_all"), key="clear_all", use_container_width=True):
            st.session_state.data_files = []
            st.session_state.data_briefing = ""
            st.session_state.paper_files = []
            st.session_state.analysis_method = ""
            st.session_state.qa_history = []
            st.session_state.message_history = []
            st.session_state.chat_history = []
            st.session_state.analysis_started = False
            st.session_state.should_run_agent = False
            st.session_state.is_streaming = False
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
            ### How to use:

            1. **Upload Data** (Panel 1)
               - Upload CSV, Excel, TSV, or gzipped files
               - Click "Analyze Data" to generate an automatic briefing

            2. **Upload Paper / Workflow** (Panel 2)
               - Upload a research paper (PDF) or paste your analysis steps
               - Click "Extract Analysis Workflow" to let the AI summarize the procedure

            3. **Start the AI Conversation** (Panel 3)
               - Click **Start Analysis** to open the chat
               - Ask the assistant to run analyses, generate plots, or refine results
               - Provide feedback in natural language (e.g., "Filter log2FC > 1", "Add a volcano plot")

            ### Tips:
            - Upload multiple files if your workflow needs them
            - Refer to generated plot names when asking for explanations or tweaks
            - Use the Q&A panel to keep track of important clarifications
            - All figures and intermediate files are saved in the workspace automatically
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
# CONVERSATIONAL ANALYSIS EXPERIENCE
# =============================================================================


def render_analysis_conversation() -> None:
    """Display the conversational analysis interface."""
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("analysis_started", False)
    st.session_state.setdefault("should_run_agent", False)
    st.session_state.setdefault("is_streaming", False)

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

    if not st.session_state.analysis_started:
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

    if not st.session_state.should_run_agent:
        return

    original_dir = os.getcwd()
    os.chdir(st.session_state.work_dir)
    try:
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
                updated_files - baseline_files,
                key=lambda path: os.path.getmtime(path),
            )
            attachments = new_files
            if result_text:
                message_placeholder.markdown(
                    format_agent_output_for_display(result_text)
                )
            if new_files:
                display_chat_files(new_files)
    finally:
        os.chdir(original_dir)

    st.session_state.should_run_agent = False
    if result_text.strip():
        maybe_add_assistant_message(result_text, files=attachments)
        _log_text_preview("Chat response", result_text)
    st.rerun()


def get_analysis_context(max_messages: int = 5) -> str:
    """Build a lightweight context string for Q&A."""
    context_parts: list[str] = []

    if st.session_state.data_briefing:
        context_parts.append("=== DATA BRIEFING ===")
        context_parts.append(st.session_state.data_briefing[:1000])

    if st.session_state.analysis_method:
        context_parts.append("=== ANALYSIS WORKFLOW ===")
        context_parts.append(st.session_state.analysis_method[:1000])

    assistant_messages = [
        msg["content"]
        for msg in st.session_state.chat_history
        if msg.get("role") == "assistant" and msg.get("content")
    ]
    if assistant_messages:
        context_parts.append("=== RECENT ANALYSIS RESPONSES ===")
        for content in assistant_messages[-max_messages:]:
            context_parts.append(content.strip())

    return "\n\n".join(context_parts) if context_parts else "No analysis context available yet."


def answer_qa_question(question):
    """Answer a Q&A question based on current analysis context"""

    context = get_analysis_context()

    prompt = f"""You are a helpful bioinformatics analysis assistant. A user is asking a question about their ongoing analysis.

ANALYSIS CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Provide a clear, concise answer based on the analysis context
- If the information is not available in the context, say so politely
- Reference specific results when relevant
- Be technical but understandable
- If the user asks \"why\", provide reasoning based on the analysis

Answer the question:"""

    try:
        llm = st.session_state.agent.llm
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        return (
            f"Error generating answer: {str(e)}\n\nPlease try rephrasing your question."
        )


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # When run directly (not from LIMS)
    run_omicshorizon_app(from_lims=False, workspace_path=None)
