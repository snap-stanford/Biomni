import streamlit as st
import os
import yaml
import re
import pandas as pd
import gzip
from collections import defaultdict
from biomni.agent import A1_HITS
from langchain_core.messages import HumanMessage, AIMessage
from biomni.config import default_config
import tempfile
import shutil
from datetime import datetime
import io
import base64
from PIL import Image
import glob

# Configuration
LLM_MODEL = "gemini-2.5-pro"
# BIOMNI_DATA_PATH = "/workdir_efs/jhjeon/Biomni/biomni_data"
# WORK_DIR = "/workdir_efs/jhjeon/Biomni/streamlit_workspace"
CURRENT_ABS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_config_paths():
    """Load path settings from project-level config.yaml if present."""
    try:
        # project root: .../Biomni_HITS/streamlit/ -> .../Biomni_HITS -> repo root
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        config_path = os.path.join(repo_root, "config.yaml")
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}
    except Exception:
        cfg = {}
    # Fallback to existing defaults if keys are missing
    biomni_data_default = "/workdir_efs/jaechang/work2/biomni_hits_test/biomni_data"
    workspace_default = "/workdir_efs/jaechang/work2/biomni_hits_test/Biomni_HITS/streamlit/streamlit_workspace"
    return (
        cfg.get("BIOMNI_DATA_PATH", biomni_data_default),
        cfg.get("WORKSPACE_PATH", workspace_default),
    )


BIOMNI_DATA_PATH, WORKSPACE_PATH = _load_config_paths()

# For this app, keep local working directory naming as before; if WORKSPACE_PATH
# points to a directory, we use it directly as our WORK_DIR. If it's a base
# workspace root, you can set it in config.yaml accordingly.
WORK_DIR = WORKSPACE_PATH

# Logo paths
LOGO_COLOR_PATH = f"{CURRENT_ABS_DIR}/logo/OMICS-HORIZON_Logo_Color.svg"
LOGO_MONO_PATH = f"{CURRENT_ABS_DIR}/logo/OMICS-HORIZON_Logo_Mono.svg"

# Constants for data processing
MAX_DATA_COLUMNS_TO_SHOW = 20
MAX_SAMPLE_EXAMPLES = 5
MIN_COLUMN_PATTERN_LENGTH = 3
MAX_CONTENT_LENGTH_FOR_LLM = 15000
MAX_DISPLAY_TEXT_LENGTH = 8000
MAX_OBSERVATION_DISPLAY_LENGTH = 2000
MIN_MEANINGFUL_CONTENT_LENGTH = 50

# Multilingual support
TRANSLATIONS = {
    "en": {
        "app_title": "OmicsHorizon™-Transcriptome",
        "app_subtitle": "AI-Powered Transcriptomic Analysis Platform",
        "panel1_title": "📊 Data Upload & Briefing",
        "panel2_title": "📄 Paper Upload & Workflow Extraction",
        "panel3_title": "Integrated Analysis",
        "upload_data": "Upload your data files",
        "upload_paper": "Upload research paper (PDF)",
        "analyze_data": "🔍 Analyze Data",
        "extract_workflow": "🔬 Extract Analysis Workflow",
        "execute_analysis": "▶️ Execute Analysis",
        "data_briefing": "📋 Data Briefing",
        "analysis_workflow": "📋 Analysis Workflow",
        "analysis_results": "Analysis Results",
        "step_by_step": "📊 Step-by-Step Results",
        "full_report": "📝 Full Report",
        "raw_output": "🔍 Raw Output",
        "control_panel": "🎛️ Control Panel",
        "session_info": "📊 Session Info",
        "clear_all": "🗑️ Clear All Data",
        "instructions": "📖 Instructions",
        "language": "🌐 Language",
        "qa_title": "💬 Analysis Q&A",
        "qa_ask_questions": "💡 Ask Questions",
        "qa_placeholder": "e.g., Why was this threshold chosen? What does the p-value mean?",
        "qa_no_analysis": "💡 Complete at least one analysis step to start asking questions",
        "qa_caption": "Ask questions about your analysis, methods, or results",
        "refinement_title": "🔧 Analysis Refinement",
        "refinement_expander": "🎯 Refine Analysis Results",
        "refinement_desc": "Make targeted modifications to your completed analysis:",
        "refinement_examples": "e.g., 'Change the volcano plot colors', 'Add sample labels to the heatmap', 'Use different statistical test'",
        "refinement_placeholder": "Describe your refinement request:",
        "refinement_example": "Example: Change the x-axis label on the PCA plot to 'Principal Component 1' or use FDR correction instead of Bonferroni for p-values",
        "refinement_plan_button": "💡 Get Refinement Plan",
        "refinement_apply_button": "⚡ Apply Refinement",
        "refinement_target_step": "Target specific step (optional):",
        "refinement_plan_title": "📋 Refinement Plan",
        "refinement_instructions_title": "Analysis Refinement (After Completion):",
        "refinement_instructions_1": "**Get Refinement Plan**: Describe what you want to change and get AI suggestions",
        "refinement_instructions_2": "**Apply Refinement**: Make targeted modifications without re-running everything",
        "refinement_instructions_3": "**Target Specific Steps**: Modify individual analysis steps as needed",
        "refinement_instructions_4": "**Examples**: Change plot labels, adjust parameters, add annotations",
        "sequential_mode": "🔄 Sequential Mode (Recommended)",
        "batch_mode": "📦 Batch Mode (All steps visible)",
        "choose_interaction_mode": "Choose interaction mode:",
        "switch_mode": "🔄 Switch Mode",
        "batch_mode_desc": "📦 **Batch Mode**: All steps visible at once.\n- ▶️ Start any step when ready\n- 💬 Provide feedback to refine results\n- 🔄 Re-run steps as needed\n- Each step builds on previous results",
        "sequential_mode_desc": "🔄 **Sequential Mode**: Step-by-step guided analysis.\n- Focus on one step at a time\n- Provide feedback after each step\n- Continue when satisfied with results",
        "ready_to_start": "🚀 Ready to Start Analysis",
        "total_steps": "Total Steps:",
        "workflow_overview": "Workflow Overview:",
        "start_analysis": "▶️ Start Analysis",
        "step_completed": "✅ Step {step_num} Completed: {step_title}",
        "step_execution": "🔬 Step {step_num}: {step_title}",
        "previous_steps_summary": "📋 Previous Steps Summary",
        "execute_step": "⚙️ Execute Step {step_num}",
        "step_feedback": "💬 Step Feedback",
        "step_feedback_placeholder": "How was Step {step_num}? Any modifications needed?",
        "step_feedback_example": "e.g., 'Change the plot colors', 'Use different parameters', 'Looks good - continue'",
        "modify_step": "🔄 Modify Step",
        "continue_to_next": "✅ Continue to Next",
        "back_to_previous": "⬅️ Back to Previous",
        "workflow_completed": "🎉 Analysis Workflow Completed!",
        "workflow_summary": "📋 Workflow Summary",
        "restart_workflow": "🔄 Restart Workflow",
        "export_results": "📦 Export Results",
        "review_steps": "⬅️ Review Steps",
    },
    "ko": {
        "app_title": "OmicsHorizon™-Transcriptome",
        "app_subtitle": "AI 기반 전사체 분석 플랫폼",
        "panel1_title": "📊 데이터 업로드 및 브리핑",
        "panel2_title": "📄 논문 업로드 및 워크플로우 추출",
        "panel3_title": "통합 분석",
        "upload_data": "데이터 파일을 업로드하세요",
        "upload_paper": "연구 논문 업로드 (PDF)",
        "analyze_data": "🔍 데이터 분석",
        "extract_workflow": "🔬 워크플로우 추출",
        "execute_analysis": "▶️ 분석 실행",
        "data_briefing": "📋 데이터 브리핑",
        "analysis_workflow": "📋 분석 워크플로우",
        "analysis_results": "분석 결과",
        "step_by_step": "📊 단계별 결과",
        "full_report": "📝 전체 보고서",
        "raw_output": "🔍 원본 출력",
        "control_panel": "🎛️ 제어판",
        "session_info": "📊 세션 정보",
        "clear_all": "🗑️ 모든 데이터 삭제",
        "instructions": "📖 사용 방법",
        "language": "🌐 언어",
        "qa_title": "💬 분석 질의응답",
        "qa_ask_questions": "💡 질문하기",
        "qa_placeholder": "예: 왜 이 임계값이 선택되었나요? p-value는 무엇을 의미하나요?",
        "qa_no_analysis": "💡 질문하려면 최소 하나의 분석 단계를 완료하세요",
        "qa_caption": "분석, 방법론, 결과에 대해 질문하세요",
        "refinement_title": "🔧 분석 정제",
        "refinement_expander": "🎯 분석 결과 정제",
        "refinement_desc": "완료된 분석에 대해 세부적인 수정을 수행하세요:",
        "refinement_examples": "예: 'volcano plot 색상 변경', 'heatmap에 샘플 라벨 추가', '다른 통계 검정 사용'",
        "refinement_placeholder": "수정 요청을 설명하세요:",
        "refinement_example": "예: PCA plot의 x축 라벨을 '주성분 1'으로 변경하거나 Bonferroni 대신 FDR 보정 사용",
        "refinement_plan_button": "💡 정제 계획 얻기",
        "refinement_apply_button": "⚡ 정제 적용",
        "refinement_target_step": "특정 단계 대상 (선택사항):",
        "refinement_plan_title": "📋 정제 계획",
        "refinement_instructions_title": "분석 정제 (완료 후):",
        "refinement_instructions_1": "**정제 계획 얻기**: 변경하고 싶은 내용을 설명하고 AI 제안 받기",
        "refinement_instructions_2": "**정제 적용**: 전체 재실행 없이 대상 수정 수행",
        "refinement_instructions_3": "**특정 단계 대상**: 개별 분석 단계 필요에 따라 수정",
        "refinement_instructions_4": "**예시**: 플롯 라벨 변경, 파라미터 조정, 주석 추가",
        "sequential_mode": "🔄 순차 모드 (권장)",
        "batch_mode": "📦 배치 모드 (모든 단계 표시)",
        "choose_interaction_mode": "상호작용 모드 선택:",
        "switch_mode": "🔄 모드 전환",
        "batch_mode_desc": "📦 **배치 모드**: 모든 단계를 한 번에 표시.\n- ▶️ 원하는 단계부터 시작\n- 💬 결과 수정 피드백 제공\n- 🔄 단계 재실행 가능\n- 이전 결과 활용",
        "sequential_mode_desc": "🔄 **순차 모드**: 단계별 안내 분석.\n- 한 단계씩 집중\n- 각 단계 후 피드백 제공\n- 만족 시 다음 단계 진행",
        "ready_to_start": "🚀 분석 시작 준비됨",
        "total_steps": "총 단계 수:",
        "workflow_overview": "워크플로우 개요:",
        "start_analysis": "▶️ 분석 시작",
        "step_completed": "✅ {step_num}단계 완료: {step_title}",
        "step_execution": "🔬 {step_num}단계: {step_title}",
        "previous_steps_summary": "📋 이전 단계 요약",
        "execute_step": "⚙️ {step_num}단계 실행",
        "step_feedback": "💬 단계 피드백",
        "step_feedback_placeholder": "{step_num}단계는 어떠셨나요? 수정이 필요한가요?",
        "step_feedback_example": "예: '플롯 색상 변경', '다른 파라미터 사용', '괜찮음 - 계속 진행'",
        "modify_step": "🔄 단계 수정",
        "continue_to_next": "✅ 다음으로 계속",
        "back_to_previous": "⬅️ 이전으로 돌아가기",
        "workflow_completed": "🎉 분석 워크플로우 완료!",
        "workflow_summary": "📋 워크플로우 요약",
        "restart_workflow": "🔄 워크플로우 재시작",
        "export_results": "📦 결과 내보내기",
        "review_steps": "⬅️ 단계 검토",
    },
}

# Page config
st.set_page_config(
    page_title="OmicsHorizon™-Transcriptome",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Cache logo loading to avoid repeated file I/O
@st.cache_data
def load_logo_base64(logo_path):
    """Load and cache logo as base64 string."""
    try:
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        st.warning(f"Logo file not found: {logo_path}")
        return ""


# Load logos once (cached)
LOGO_COLOR_BASE64 = load_logo_base64(LOGO_COLOR_PATH)
LOGO_MONO_BASE64 = load_logo_base64(LOGO_MONO_PATH)

# Custom CSS for better layout and logo theming
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .panel-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .stTextArea textarea {
        font-family: monospace;
    }
    div[data-testid="stExpander"] {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
    }
    
    /* Wider sidebar for Q&A */
    [data-testid="stSidebar"] {
        min-width: 420px !important;
        max-width: 420px !important;
    }
    
    /* Logo container styling */
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1rem 0;
        margin-bottom: -2rem;
    }
    .logo-container img {
        max-width: 100%;
        height: auto;
    }
    
    /* Light mode: show color logo, hide mono logo */
    .logo-light {
        display: block !important;
    }
    .logo-dark {
        display: none !important;
    }
    
    /* Dark mode: hide color logo, show mono logo */
    /* Multiple selectors for robust theme detection */
    @media (prefers-color-scheme: dark) {
        .logo-light {
            display: none !important;
        }
        .logo-dark {
            display: block !important;
        }
    }
    
    /* Streamlit theme detection - multiple approaches */
    [data-theme="dark"] .logo-light,
    [data-baseweb-theme="dark"] .logo-light,
    .stApp[data-theme="dark"] .logo-light {
        display: none !important;
    }
    
    [data-theme="dark"] .logo-dark,
    [data-baseweb-theme="dark"] .logo-dark,
    .stApp[data-theme="dark"] .logo-dark {
        display: block !important;
    }
    
    /* Main page logo styling */
    .main-logo {
        margin: 0 auto;
        position: relative;
    }
    
    /* Ensure proper z-index stacking */
    .logo-light {
        z-index: 2;
    }
    .logo-dark {
        z-index: 1;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Add logo to sidebar
with st.sidebar:
    # Display both logos with CSS-based theme switching (SVG)
    if LOGO_COLOR_BASE64 and LOGO_MONO_BASE64:
        st.markdown(
            f"""
        <div class="logo-container">
            <img src="data:image/svg+xml;base64,{LOGO_COLOR_BASE64}" 
                 class="logo-light" alt="OMICS-HORIZON Logo">
            <img src="data:image/svg+xml;base64,{LOGO_MONO_BASE64}" 
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


# Initialize session state
if "language" not in st.session_state:
    st.session_state.language = "en"  # Default language

if "agent" not in st.session_state:
    default_config.llm = LLM_MODEL
    default_config.commercial_mode = True
    st.session_state.agent = A1_HITS(
        path=BIOMNI_DATA_PATH,
        llm=LLM_MODEL,
        use_tool_retriever=True,
    )

if "data_files" not in st.session_state:
    st.session_state.data_files = []
if "data_briefing" not in st.session_state:
    st.session_state.data_briefing = ""
if "paper_files" not in st.session_state:
    st.session_state.paper_files = []
if "analysis_method" not in st.session_state:
    st.session_state.analysis_method = ""
if "message_history" not in st.session_state:
    st.session_state.message_history = []
if "work_dir" not in st.session_state:
    # Create unique work directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.work_dir = os.path.join(WORK_DIR, f"session_{timestamp}")
    os.makedirs(st.session_state.work_dir, exist_ok=True)
    # Don't change directory - use absolute paths instead

# Interactive mode: Step-by-step state management
if "steps_state" not in st.session_state:
    st.session_state.steps_state = {}  # {step_num: {...}}
if "current_step" not in st.session_state:
    st.session_state.current_step = 0  # 0 = not started

# Q&A state management
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []  # Global Q&A chat history


def format_agent_output_for_display(raw_text):
    """Format agent's raw output into clean, readable Markdown.

    Converts:
    - <execute>code</execute> → Python/R/Bash code blocks with syntax highlighting
    - <observation>result</observation> → Formatted result boxes
    - <solution>answer</solution> → Final answer boxes
    - Adds section dividers for better visual separation
    - Handles incomplete tags during streaming

    Args:
        raw_text: Raw agent output with XML-like tags

    Returns:
        Formatted markdown string
    """
    formatted = raw_text

    # Step 0: Handle incomplete <execute> tags (streaming in progress)
    # Find incomplete execute blocks and temporarily format them
    incomplete_execute = re.search(
        r"<execute>((?:(?!<execute>|</execute>).)*?)$", formatted, re.DOTALL
    )
    incomplete_code = None

    if incomplete_execute:
        # Extract the incomplete code
        incomplete_code = incomplete_execute.group(1)
        # Remove it temporarily (we'll add it back formatted at the end)
        formatted = formatted[: incomplete_execute.start()]

    # Step 1: Convert complete <execute> tags to code blocks
    execution_count = [0]  # Use list to maintain state in nested function

    def replace_execute_block(match):
        code = match.group(1).strip()
        execution_count[0] += 1

        # Detect language
        if code.startswith("#!R"):
            language = "r"
            code = code[3:].strip()  # Remove #!R marker
            lang_emoji = "📊"
        elif code.startswith("#!BASH"):
            language = "bash"
            code = code[6:].strip()  # Remove #!BASH marker
            lang_emoji = "🔧"
        else:
            language = "python"
            lang_emoji = "🐍"

        # Return formatted code block with header and divider
        return f"\n\n---\n\n{lang_emoji} **코드 실행 #{execution_count[0]}:**\n```{language}\n{code}\n```\n"

    formatted = re.sub(
        r"<execute>\s*(.*?)\s*</execute>",
        replace_execute_block,
        formatted,
        flags=re.DOTALL,
    )

    # Step 2: Convert <observation> tags to result boxes
    def replace_observation_block(match):
        result = match.group(1).strip()

        # Check if it's an error
        is_error = any(
            keyword in result
            for keyword in ["Error", "Exception", "Traceback", "Failed"]
        )

        if is_error:
            # Error format - use error box style
            return f"\n\n❌ **실행 오류:**\n```\n{result}\n```\n"
        else:
            # Success format - use success box style
            # If result is very long, truncate for display
            if len(result) > MAX_OBSERVATION_DISPLAY_LENGTH:
                result_preview = result[:MAX_OBSERVATION_DISPLAY_LENGTH]
                result = (
                    result_preview
                    + "\n\n... (출력이 길어 생략됨. 총 "
                    + str(len(result))
                    + "자)"
                )

            return f"\n\n✅ **실행 성공:**\n```\n{result}\n```\n"

    formatted = re.sub(
        r"<observation>\s*(.*?)\s*</observation>",
        replace_observation_block,
        formatted,
        flags=re.DOTALL,
    )

    # Step 3: Convert <solution> tags to final answer boxes
    def replace_solution_block(match):
        solution = match.group(1).strip()
        return f"\n\n---\n\n🎯 **최종 답변:**\n\n{solution}\n\n---\n"

    formatted = re.sub(
        r"<solution>\s*(.*?)\s*</solution>",
        replace_solution_block,
        formatted,
        flags=re.DOTALL,
    )

    # Step 4: Add visual markers to plan checkboxes for better visibility
    # Enhance completed items
    formatted = re.sub(
        r"^(\s*\d+\.\s*)\[✓\](.+?)(?:\(completed\))?$",
        r"\1✅ \2",
        formatted,
        flags=re.MULTILINE,
    )

    # Enhance failed items
    formatted = re.sub(
        r"^(\s*\d+\.\s*)\[✗\](.+?)$", r"\1❌ \2", formatted, flags=re.MULTILINE
    )

    # Enhance pending items
    formatted = re.sub(
        r"^(\s*\d+\.\s*)\[\s\](.+?)$", r"\1⬜ \2", formatted, flags=re.MULTILINE
    )

    # Step 5: Clean up excessive whitespace
    # Replace 3+ consecutive newlines with just 2
    formatted = re.sub(r"\n{3,}", "\n\n", formatted)

    # Step 6: Handle incomplete code from streaming
    if incomplete_code:
        # Detect language for incomplete code
        if incomplete_code.strip().startswith("#!R"):
            language = "r"
            code_content = incomplete_code[3:].strip()
            lang_emoji = "📊"
        elif incomplete_code.strip().startswith("#!BASH"):
            language = "bash"
            code_content = incomplete_code[6:].strip()
            lang_emoji = "🔧"
        else:
            language = "python"
            code_content = incomplete_code.strip()
            lang_emoji = "🐍"

        # Add the incomplete code as a formatted block with "실행 중..." indicator
        execution_count[0] += 1
        formatted += f"\n\n---\n\n{lang_emoji} **코드 실행 #{execution_count[0]}:** ⏳ 실행 중...\n```{language}\n{code_content}\n```\n"

    # Step 7: Handle incomplete observation tags
    incomplete_obs = re.search(
        r"<observation>((?:(?!<observation>|</observation>).)*?)$", formatted, re.DOTALL
    )
    if incomplete_obs:
        obs_content = incomplete_obs.group(1).strip()
        formatted = formatted[: incomplete_obs.start()]
        if obs_content:
            formatted += f"\n\n⏳ **실행 중...**\n```\n{obs_content}\n```\n"

    # Step 8: Protect Python comments from being interpreted as Markdown headers
    # Replace standalone lines starting with # (Python comments) to prevent Markdown header interpretation
    # This is applied to the final text to protect comments that aren't in code blocks

    # Find all code blocks first to avoid modifying them
    code_blocks = []

    def save_code_block(match):
        idx = len(code_blocks)
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{idx}__"

    # Temporarily replace code blocks with placeholders
    formatted = re.sub(r"```[\s\S]*?```", save_code_block, formatted)

    # Now protect Python comments outside of code blocks
    # Convert lines starting with # to escaped format
    lines = formatted.split("\n")
    protected_lines = []
    for line in lines:
        stripped = line.lstrip()
        # If line starts with # but is not a Markdown header (check for single # followed by space)
        if stripped.startswith("#") and not stripped.startswith("##"):
            # Check if it looks like a comment (has more than just #)
            if len(stripped) > 1 and not stripped[1:].strip().startswith("#"):
                # Add a zero-width space to prevent Markdown interpretation
                indent = len(line) - len(stripped)
                protected_lines.append(" " * indent + "\\# " + stripped[1:])
            else:
                protected_lines.append(line)
        else:
            protected_lines.append(line)
    formatted = "\n".join(protected_lines)

    # Restore code blocks
    for idx, code_block in enumerate(code_blocks):
        formatted = formatted.replace(f"__CODE_BLOCK_{idx}__", code_block)

    return formatted


def parse_step_progress(accumulated_text):
    """Parse current step progress from agent output.

    Returns:
        dict with keys: total_steps, completed_steps, current_step_num, current_step_title
    """
    # Pattern 1: Count checkboxes to determine total steps
    # Also handle enhanced checkboxes with emojis (✅, ⬜, ❌)
    all_checkboxes = re.findall(
        r"^\s*(\d+)\.\s*(?:\[([✓✗ ])\]|([✅❌⬜]))\s*(.+?)(?:\s*\(.*?\))?$",
        accumulated_text,
        re.MULTILINE,
    )

    # Parse checkbox status and store in dict to handle duplicates
    # (Agent may output plan multiple times during updates)
    checkbox_dict = {}  # {step_num: {'status': ..., 'title': ...}}

    for match in all_checkboxes:
        num_str = match[0]
        old_status = match[1]  # [✓], [✗], [ ]
        emoji_status = match[2]  # ✅, ❌, ⬜
        title = match[3]

        # Determine status
        if old_status == "✓" or emoji_status == "✅":
            status = "completed"
        elif old_status == "✗" or emoji_status == "❌":
            status = "failed"
        else:  # ' ' or ⬜
            status = "pending"

        step_num = int(num_str)

        # Update dict (later occurrences override earlier ones)
        checkbox_dict[step_num] = {"status": status, "title": title.strip()}

    # Convert dict to sorted list
    parsed_checkboxes = [
        {"num": num, "status": data["status"], "title": data["title"]}
        for num, data in sorted(checkbox_dict.items())
    ]

    total_steps = len(parsed_checkboxes)
    completed_steps = sum(1 for cb in parsed_checkboxes if cb["status"] == "completed")

    # Pattern 2: Find current step being executed (our marker format)
    current_marker = re.search(
        r"===\s*Step\s+(\d+)[:\s]+([^=]+?)===", accumulated_text, re.IGNORECASE
    )

    current_step_num = None
    current_step_title = None

    if current_marker:
        current_step_num = int(current_marker.group(1))
        current_step_title = current_marker.group(2).strip()
    else:
        # Fallback logic: Find current step from checkboxes
        if total_steps > 0 and parsed_checkboxes:
            # Strategy 1: Find the last completed step, then get next one
            last_completed_num = 0
            for cb in parsed_checkboxes:
                if cb["status"] == "completed":
                    last_completed_num = max(last_completed_num, cb["num"])

            # Current step is the one right after last completed
            if last_completed_num > 0:
                # Find the next step after last_completed_num
                next_steps = [
                    cb for cb in parsed_checkboxes if cb["num"] > last_completed_num
                ]
                if next_steps:
                    next_steps.sort(key=lambda x: x["num"])
                    current_step_num = next_steps[0]["num"]
                    current_step_title = next_steps[0]["title"]

            # Strategy 2: If no completed steps yet, use first pending
            if current_step_num is None:
                for cb in parsed_checkboxes:
                    if cb["status"] == "pending":
                        current_step_num = cb["num"]
                        current_step_title = cb["title"]
                        break

            # Strategy 3: If still not found, use completed_steps + 1
            if current_step_num is None:
                next_step_num = completed_steps + 1
                if next_step_num <= total_steps:
                    for cb in parsed_checkboxes:
                        if cb["num"] == next_step_num:
                            current_step_num = next_step_num
                            current_step_title = cb["title"]
                            break

    # Pattern 3: Detect if agent is thinking/executing
    is_executing = bool(
        re.search(r"<execute>(?!.*</execute>)", accumulated_text, re.DOTALL)
    )
    is_thinking = bool(
        re.search(
            r"(?:thinking|analyzing|processing)", accumulated_text[-500:], re.IGNORECASE
        )
    )

    return {
        "total_steps": total_steps,
        "completed_steps": completed_steps,
        "current_step_num": current_step_num,
        "current_step_title": current_step_title,
        "is_executing": is_executing,
        "is_thinking": is_thinking,
    }


def process_with_agent(prompt: str, show_process=False, use_history=False):
    """Process prompt with Biomni agent.

    Args:
        prompt: User prompt
        show_process: Whether to show analysis process in expander
        use_history: If True, include conversation history. If False, only use current prompt.
    """
    # Change to work directory for agent execution
    original_dir = os.getcwd()
    try:
        os.chdir(st.session_state.work_dir)

        # Convert to agent format
        if use_history:
            # Include full conversation history
            st.session_state.message_history.append({"role": "user", "content": prompt})
            agent_input = []
            for msg in st.session_state.message_history:
                if msg["role"] == "user":
                    agent_input.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    agent_input.append(AIMessage(content=msg["content"]))
        else:
            # Only use current prompt (no history)
            agent_input = [HumanMessage(content=prompt)]

        # Create expander for process if requested
        if show_process:
            # Progress UI elements
            progress_bar = st.progress(0)
            status_container = st.empty()
            step_info_container = st.empty()

            with st.expander("🔍 View Analysis Process", expanded=False):
                process_container = st.empty()
                result = ""
                message_stream = st.session_state.agent.go_stream(agent_input)

                for chunk in message_stream:
                    node = chunk[1][1]["langgraph_node"]
                    chunk_data = chunk[1][0]

                    if node == "generate" and hasattr(chunk_data, "content"):
                        result += chunk_data.content

                        # Format the output for better readability
                        formatted_result = format_agent_output_for_display(result)
                        process_container.markdown(formatted_result)

                        # Update progress indicators
                        progress_info = parse_step_progress(result)

                        # Update progress bar
                        if progress_info["total_steps"] > 0:
                            progress = (
                                progress_info["completed_steps"]
                                / progress_info["total_steps"]
                            )
                            progress_bar.progress(
                                min(progress, 0.99)
                            )  # Cap at 99% until fully done

                            # Update status
                            completed = progress_info["completed_steps"]
                            total = progress_info["total_steps"]

                            if (
                                progress_info["current_step_num"]
                                and progress_info["current_step_title"]
                            ):
                                status_emoji = (
                                    "⚙️" if progress_info["is_executing"] else "🧠"
                                )
                                step_info_container.markdown(
                                    f"{status_emoji} **In Progress: Step {progress_info['current_step_num']}/{total}** - {progress_info['current_step_title']}"
                                )

                            status_container.info(
                                f"✅ Completed: {completed}/{total} steps | ⏳ In Progress: {total - completed} steps"
                            )
                        else:
                            status_container.info("🔍 Planning analysis steps...")

                    elif node == "execute" and hasattr(chunk_data, "content"):
                        result += chunk_data.content

                        # Format the output for better readability
                        formatted_result = format_agent_output_for_display(result)
                        process_container.markdown(formatted_result)

                        # Update progress for execution
                        progress_info = parse_step_progress(result)
                        if progress_info["total_steps"] > 0:
                            progress = (
                                progress_info["completed_steps"]
                                / progress_info["total_steps"]
                            )
                            progress_bar.progress(min(progress, 0.99))

                # Mark as complete
                progress_bar.progress(1.0)
                status_container.success("✅ Analysis complete!")
                step_info_container.empty()

                # Save formatted process to session state for later display
                final_formatted = format_agent_output_for_display(result)
                st.session_state.analysis_process = final_formatted
        else:
            # Silent processing with enhanced progress tracking
            result = ""
            progress_bar = st.progress(0)
            status_container = st.empty()
            step_info_container = st.empty()

            message_stream = st.session_state.agent.go_stream(agent_input)

            for chunk in message_stream:
                node = chunk[1][1]["langgraph_node"]
                chunk_data = chunk[1][0]

                if node == "generate" and hasattr(chunk_data, "content"):
                    result += chunk_data.content
                elif node == "execute" and hasattr(chunk_data, "content"):
                    result += chunk_data.content

                # Update progress indicators
                progress_info = parse_step_progress(result)

                if progress_info["total_steps"] > 0:
                    # Calculate and update progress
                    progress = (
                        progress_info["completed_steps"] / progress_info["total_steps"]
                    )
                    progress_bar.progress(min(progress, 0.99))

                    # Show current step info
                    completed = progress_info["completed_steps"]
                    total = progress_info["total_steps"]

                    if (
                        progress_info["current_step_num"]
                        and progress_info["current_step_title"]
                    ):
                        status_emoji = "⚙️" if progress_info["is_executing"] else "🧠"
                        step_info_container.markdown(
                            f"{status_emoji} **In Progress: Step {progress_info['current_step_num']}/{total}** - {progress_info['current_step_title']}"
                        )

                    status_container.info(
                        f"✅ Completed: {completed}/{total} steps | ⏳ In Progress: {total - completed} steps"
                    )
                else:
                    # Still planning
                    progress_bar.progress(0.05)
                    status_container.info("🔍 Planning analysis steps...")
                    step_info_container.empty()

            # Mark as complete
            progress_bar.progress(1.0)
            status_container.success("✅ Analysis complete!")
            step_info_container.empty()

            # Clean up after a brief moment
            import time

            time.sleep(0.5)
            progress_bar.empty()
            status_container.empty()

        # Only save to history if use_history is True
        if use_history:
            st.session_state.message_history.append(
                {"role": "assistant", "content": result}
            )
        return result
    finally:
        # Always return to original directory
        os.chdir(original_dir)


def save_uploaded_file(uploaded_file):
    """Save uploaded file to work directory."""
    file_path = os.path.join(st.session_state.work_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
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

    # Page config: Only set when running standalone to avoid duplicate config and layout clashes
    if not from_lims:
        st.set_page_config(
            page_title="OmicsHorizon™-Transcriptome",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    # Cache logo loading to avoid repeated file I/O
    @st.cache_data
    def load_logo_base64(logo_path):
        """Load and cache logo as base64 string."""
        try:
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except FileNotFoundError:
            st.warning(f"Logo file not found: {logo_path}")
            return ""

    # Load logos once (cached)
    LOGO_COLOR_BASE64 = load_logo_base64(LOGO_COLOR_PATH)
    LOGO_MONO_BASE64 = load_logo_base64(LOGO_MONO_PATH)

    # Custom CSS for better layout and logo theming
    sidebar_width_rule = (
        """
        [data-testid=\"stSidebar\"] {
            min-width: 420px !important;
            max-width: 420px !important;
        }
        """
        if not from_lims
        else ""
    )

    css_block = """
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .panel-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
            padding: 10px;
            background-color: #f0f2f6;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .stTextArea textarea {
            font-family: monospace;
        }
        div[data-testid="stExpander"] {
            border: 2px solid #e0e0e0;
            border-radius: 10px;
        }

        /* Wider sidebar for Q&A (only when running standalone) */
        %SIDEBAR_WIDTH_RULE%

        /* Logo container styling */
        .logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 1rem 0;
            margin-bottom: -2rem;
        }
        .logo-container img {
            max-width: 100%;
            height: auto;
        }

        /* Light mode: show color logo, hide mono logo */
        .logo-light {
            display: block !important;
        }
        .logo-dark {
            display: none !important;
        }

        /* Dark mode: hide color logo, show mono logo */
        /* Multiple selectors for robust theme detection */
        @media (prefers-color-scheme: dark) {
            .logo-light {
                display: none !important;
            }
            .logo-dark {
                display: block !important;
            }
        }

        /* Streamlit theme detection - multiple approaches */
        [data-theme="dark"] .logo-light,
        [data-baseweb-theme="dark"] .logo-light,
        .stApp[data-theme="dark"] .logo-light {
            display: none !important;
        }

        [data-theme="dark"] .logo-dark,
        [data-baseweb-theme="dark"] .logo-dark,
        .stApp[data-theme="dark"] .logo-dark {
            display: block !important;
        }

        /* Main page logo styling */
        .main-logo {
            margin: 0 auto;
            position: relative;
        }

        /* Ensure proper z-index stacking */
        .logo-light {
            z-index: 2;
        }
        .logo-dark {
            z-index: 1;
        }
    </style>
    """.replace(
        "%SIDEBAR_WIDTH_RULE%", sidebar_width_rule
    )

    st.markdown(css_block, unsafe_allow_html=True)

    # Add logo to sidebar
    with st.sidebar:
        # Display both logos with CSS-based theme switching (SVG)
        if LOGO_COLOR_BASE64 and LOGO_MONO_BASE64:
            st.markdown(
                f"""
            <div class="logo-container">
                <img src="data:image/svg+xml;base64,{LOGO_COLOR_BASE64}"
                     class="logo-light" alt="OMICS-HORIZON Logo">
                <img src="data:image/svg+xml;base64,{LOGO_MONO_BASE64}"
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

    # Initialize session state
    if "language" not in st.session_state:
        st.session_state.language = "en"  # Default language

    if "agent" not in st.session_state:
        default_config.llm = LLM_MODEL
        default_config.commercial_mode = True
        st.session_state.agent = A1_HITS(
            path=BIOMNI_DATA_PATH,
            llm=LLM_MODEL,
            use_tool_retriever=True,
        )

    if "data_files" not in st.session_state:
        st.session_state.data_files = []
    if "data_briefing" not in st.session_state:
        st.session_state.data_briefing = ""
    if "paper_files" not in st.session_state:
        st.session_state.paper_files = []
    if "analysis_method" not in st.session_state:
        st.session_state.analysis_method = ""
    if "message_history" not in st.session_state:
        st.session_state.message_history = []
    if "work_dir" not in st.session_state:
        # Use provided workspace path if from LIMS, otherwise create unique work directory
        if from_lims and workspace_path:
            st.session_state.work_dir = workspace_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.work_dir = os.path.join(WORK_DIR, f"session_{timestamp}")
            os.makedirs(st.session_state.work_dir, exist_ok=True)
        # Don't change directory - use absolute paths instead

    # Interactive mode: Step-by-step state management
    if "steps_state" not in st.session_state:
        st.session_state.steps_state = {}  # {step_num: {...}}
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0  # 0 = not started
    if "interaction_mode" not in st.session_state:
        st.session_state.interaction_mode = "sequential"  # 'sequential' or 'batch'
    if "step_feedback" not in st.session_state:
        st.session_state.step_feedback = {}  # {step_num: feedback_text}

    # Q&A state management
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []  # Global Q&A chat history

    # Handle LIMS integration
    if "from_lims" not in st.session_state:
        st.session_state.from_lims = from_lims

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
    if LOGO_COLOR_BASE64 and LOGO_MONO_BASE64:
        st.markdown(
            f"""
        <div style="text-align: center; margin-bottom: 1rem; line-height: 0;">
            <img src="data:image/svg+xml;base64,{LOGO_COLOR_BASE64}"
                 class="logo-light main-logo" alt="OMICS-HORIZON Logo"
                 style="max-width: 600px; height: auto; margin: 0 auto;">
            <img src="data:image/svg+xml;base64,{LOGO_MONO_BASE64}"
                 class="logo-dark main-logo" alt="OMICS-HORIZON Logo"
                 style="max-width: 600px; height: auto; margin: 0 auto;">
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Title section
    # st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem; margin-top: 0.5rem; margin-bottom: 2rem;">AI-Powered Transcriptomic Analysis Platform</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Top row: 2 panels side by side
    col1, col2 = st.columns(2)

    # Panel 1: Data Upload & Briefing (Left) - Only show if not from LIMS
    if not from_lims:
        with col1:
            st.markdown(
                f'<div class="panel-header">{t("panel1_title")}</div>',
                unsafe_allow_html=True,
            )

            # File uploader
            uploaded_data = st.file_uploader(
                t("upload_data"),
                type=["csv", "xlsx", "xls", "tsv", "txt", "json", "gz"],
                accept_multiple_files=True,
                key="data_uploader",
            )

            if uploaded_data:
                # Show uploaded files
                st.info(f"📁 Uploaded {len(uploaded_data)} file(s)")
                for file in uploaded_data:
                    st.text(f"  • {file.name}")

                # Analyze button
                if st.button(t("analyze_data"), type="primary", key="analyze_data"):
                    # Save files
                    file_names = []
                    file_paths = []
                    for file in uploaded_data:
                        file_name = save_uploaded_file(file)
                        file_names.append(file_name)
                        file_paths.append(
                            os.path.join(st.session_state.work_dir, file_name)
                        )
                        if file_name not in st.session_state.data_files:
                            st.session_state.data_files.append(file_name)

                    # Direct LLM analysis without agent (no history needed)
                    with st.spinner("📊 Analyzing data files..."):
                        result = analyze_data_direct(file_paths)
                    st.session_state.data_briefing = result
                    st.rerun()

            # Display briefing
            if st.session_state.data_briefing:
                st.markdown("---")
                st.markdown(f"### {t('data_briefing')}")
                st.markdown(st.session_state.data_briefing)
    else:
        # LIMS integration: Show loaded data info
        with col1:
            st.markdown(
                '<div class="panel-header">📊 Data from LIMS</div>',
                unsafe_allow_html=True,
            )

            # Scrollable content area (render as Markdown)
            has_any_content = False
            md_parts = []

            # Selected data files from LIMS
            if (
                "selected_data_files" in st.session_state
                and st.session_state.selected_data_files
            ):
                has_any_content = True
                md_parts.append("**Selected files from LIMS:**")
                for file_info in st.session_state.selected_data_files:
                    md_parts.append(f"- {file_info['name']} ({file_info['extension']})")
                md_parts.append("")

            # Loaded data files
            if st.session_state.data_files:
                has_any_content = True
                md_parts.append(
                    f"✅ Loaded {len(st.session_state.data_files)} file(s) from LIMS:"
                )
                for filename in st.session_state.data_files:
                    md_parts.append(f"- {filename}")
                md_parts.append("")

                # Display briefing
                if st.session_state.data_briefing:
                    md_parts.append("---")
                    md_parts.append(f"### {t('data_briefing')}")
                    md_parts.append("")
                    briefing_text = (
                        st.session_state.data_briefing
                        if isinstance(st.session_state.data_briefing, str)
                        else str(st.session_state.data_briefing)
                    )
                    md_parts.append(briefing_text)
                else:
                    md_parts.append("📝 Generating data briefing...")

            if has_any_content:
                full_md = "\n".join(md_parts)
                try:
                    import markdown as _md

                    rendered_html = _md.markdown(full_md, extensions=["extra"])  # type: ignore
                    scroll_html = (
                        '<div style="max-height: 320px; overflow-y: auto; padding-right: 8px;">'
                        + rendered_html
                        + "</div>"
                    )
                    st.markdown(scroll_html, unsafe_allow_html=True)
                except Exception:
                    # Fallback: show as Markdown (without wrapper)
                    st.markdown(full_md)
            else:
                st.warning("⚠️ No data files loaded from LIMS")

    # Panel 2: Paper Upload & Method Extraction (Right)
    with col2:
        st.markdown(
            f'<div class="panel-header">{t("panel2_title")}</div>',
            unsafe_allow_html=True,
        )

        # File uploader
        uploaded_paper = st.file_uploader(
            t("upload_paper"),
            type=["pdf", "txt", "doc", "docx"],
            accept_multiple_files=False,
            key="paper_uploader",
        )

        if uploaded_paper:
            st.info(f"📄 Uploaded: {uploaded_paper.name}")

            # Extraction mode selector
            extraction_mode = st.radio(
                (
                    "추출 방식:"
                    if st.session_state.language == "ko"
                    else "Extraction Mode:"
                ),
                options=[
                    (
                        "integrated",
                        (
                            "🎯 Results + Methods (추천)"
                            if st.session_state.language == "ko"
                            else "🎯 Results + Methods (Recommended)"
                        ),
                    ),
                    (
                        "methods_only",
                        (
                            "📋 Methods만"
                            if st.session_state.language == "ko"
                            else "📋 Methods Only"
                        ),
                    ),
                    (
                        "results_only",
                        (
                            "📊 Results만"
                            if st.session_state.language == "ko"
                            else "📊 Results Only"
                        ),
                    ),
                ],
                format_func=lambda x: x[1],
                horizontal=True,
                key="extraction_mode",
                help=(
                    "• Results+Methods: 분석 순서와 세부 방법을 통합 추출\n• Methods만: 기존 방식\n• Results만: 분석 순서만 추출"
                    if st.session_state.language == "ko"
                    else "• Results+Methods: Extract analysis order and detailed methods\n• Methods only: Traditional approach\n• Results only: Extract analysis sequence only"
                ),
            )

            mode = extraction_mode[0]

            # Extract method button
            if st.button(t("extract_workflow"), type="primary", key="extract_method"):
                # Save file
                file_name = save_uploaded_file(uploaded_paper)
                if file_name not in st.session_state.paper_files:
                    st.session_state.paper_files.append(file_name)

                spinner_text = {
                    "integrated": "📖 Extracting workflow by analyzing Results and Methods sections...",
                    "methods_only": "📖 Extracting workflow from Methods section...",
                    "results_only": "📖 Extracting analysis sequence from Results section...",
                }

                with st.spinner(spinner_text[mode]):
                    # Extract workflow with selected mode
                    result = extract_workflow_from_paper(
                        os.path.join(st.session_state.work_dir, file_name), mode=mode
                    )
                    st.session_state.analysis_method = result
                st.success(f"✅ Workflow extraction complete! ({extraction_mode[1]})")
                st.rerun()

        # Scrollable content area (render as Markdown)
        md_parts = []

        # Display uploaded paper files
        if "paper_files" in st.session_state and st.session_state.paper_files:
            md_parts.append(
                f"**Uploaded Paper Files:** ({len(st.session_state.paper_files)} file(s))"
            )
            for filename in st.session_state.paper_files:
                md_parts.append(f"- {filename}")
            md_parts.append("")

        # Show uploaded files in scrollable area if no method yet
        if not st.session_state.analysis_method and md_parts:
            full_md = "\n".join(md_parts)
            try:
                import markdown as _md

                rendered_html = _md.markdown(full_md, extensions=["extra"])  # type: ignore
                scroll_html = (
                    '<div style="max-height: 400px; overflow-y: auto; padding-right: 8px;">'
                    + rendered_html
                    + "</div>"
                )
                st.markdown(scroll_html, unsafe_allow_html=True)
            except Exception:
                st.markdown(full_md)

        # Display and edit method
        if st.session_state.analysis_method:
            clean_method = st.session_state.analysis_method

            # Show method in tabs
            method_tab1, method_tab2 = st.tabs(["📋 Analysis Workflow", "✏️ Edit"])

            with method_tab1:
                method_md_parts = []

                # Add uploaded files info if exists
                if md_parts:
                    method_md_parts.extend(md_parts)

                # Add method content
                method_md_parts.append("### 🔬 Extracted Analysis Steps")
                method_md_parts.append("")

                if clean_method:
                    method_md_parts.append(clean_method)
                else:
                    method_md_parts.append(
                        "*No method extracted. Please edit to add steps.*"
                    )

                # Render scrollable content in tab1
                if method_md_parts:
                    full_md = "\n".join(method_md_parts)
                    try:
                        import markdown as _md

                        rendered_html = _md.markdown(
                            full_md, extensions=["extra"]
                        )  # type: ignore
                        scroll_html = (
                            '<div style="max-height: 400px; overflow-y: auto; padding-right: 8px;">'
                            + rendered_html
                            + "</div>"
                        )
                        st.markdown(scroll_html, unsafe_allow_html=True)
                    except Exception:
                        st.markdown(full_md)

            with method_tab2:
                st.info("💡 Format: Numbered list with tool names and parameters")

                edited_method = st.text_area(
                    "Analysis Steps",
                    value=clean_method,
                    height=500,
                    key="method_editor",
                    placeholder="1. Preprocessing: log2 transformation using tool X\n2. DEG analysis: DESeq2 with |log2FC| > 2, p < 0.01\n3. Clustering: hierarchical clustering, heatmap\n...",
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "💾 Save",
                        key="save_method",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state.analysis_method = edited_method
                        st.success("✅ Saved!")
                        st.rerun()

                with col2:
                    if st.button(
                        "🔄 Reset", key="reset_method", use_container_width=True
                    ):
                        st.rerun()

        elif st.button("✍️ Write Custom Method", key="write_custom"):
            st.session_state.analysis_method = """1. Preprocessing: describe preprocessing, mention tools
2. Quality control: filtering criteria
3. Statistical analysis: test name, parameters (e.g., p < 0.05)
4. Clustering: method, visualization
5. Enrichment analysis: tool name, database
..."""
            st.rerun()

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
        st.info(
            f"""
        - Data files: {len(st.session_state.data_files)}
        - Paper files: {len(st.session_state.paper_files)}
        - Method defined: {'✅' if st.session_state.analysis_method else '❌'}
        - Work directory: `{st.session_state.work_dir.lstrip('/workdir_efs/jhjeon/Biomni/streamlit_workspace/')}`
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
        result = process_with_agent(prompt, show_process=True, use_history=False)

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


def initialize_step_state(step_num, step_data):
    """Initialize state for a specific step if not exists"""
    if step_num not in st.session_state.steps_state:
        st.session_state.steps_state[step_num] = {
            "status": "pending",  # 'pending', 'in_progress', 'completed', 'error'
            "title": step_data["title"],
            "description": step_data.get("description", ""),
            "result": None,
            "solution": None,
            "formatted_process": None,
            "files": [],
            "feedback": None,
            "iteration": 0,
        }


@st.fragment
def render_sequential_interactive_mode(analysis_steps):
    """Render sequential interactive mode with chat-like interface"""
    total_steps = len(analysis_steps)

    # Initialize chat history if not exists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize completed processes storage if not exists
    if "completed_processes" not in st.session_state:
        st.session_state.completed_processes = {}

    # Initialize pending user input queue if not exists
    if "pending_user_inputs" not in st.session_state:
        st.session_state.pending_user_inputs = []

    # Initialize streaming state if not exists
    if "is_streaming" not in st.session_state:
        st.session_state.is_streaming = False

    # Chat-like interface container
    st.markdown("### 💬 Analysis Conversation")

    # 분석하기 버튼 추가
    if st.button(
        "Start Analysis",
        key="start_analysis_btn",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.analysis_started = True

    # 버튼이 클릭되었을 때만 아래 코드 실행
    if st.session_state.get("analysis_started", False):
        # Display initial greeting
        if not st.session_state.chat_history:
            with st.chat_message("assistant"):
                st.markdown("👋 **Hello! I'm your OmicsHorizon analysis assistant.**")

        # Display chat history - ensure it's always shown
        # This will display all previous messages every time the page reruns
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("files"):
                    display_chat_files(message["files"])

        os.chdir(st.session_state.work_dir)
        data_info = ", ".join([f"`{f}`" for f in st.session_state.data_files])

        prompt = f"""Perform bioinformatics analysis.
#Analysis Instructions:
{st.session_state.analysis_method}

DATA FILES: {data_info}

DATA BRIEFING:
{st.session_state.data_briefing if st.session_state.data_briefing else "Files are available in the working directory"}

"""
        print(prompt)

        with st.chat_message("assistant"):
            result = ""
            message_placeholder = st.empty()

            st.session_state.user_interupt_message = None
            st.session_state.agent.state
            # Spinner during streaming (indeterminate loading)
            with st.spinner("AI is performing the analysis...…"):
                user_interupt_message = st.chat_input("Enter your message...")
                if user_interupt_message:
                    st.session_history.append(
                        {"role": "user", "content": user_interupt_message}
                    )
                    with st.chat_message("user"):
                        st.markdown(user_interupt_message)
                    st.session_state.agent.state["messages"].append(
                        HumanMessage(content=user_interupt_message)
                    )
                agent_input = build_agent_input_from_history(
                    initial_prompt=prompt, include_initial=True
                )
                message_stream = st.session_state.agent.go_stream(agent_input)

                count = 0
                prev_node = None
                for chunk in message_stream:
                    node = chunk[1][1]["langgraph_node"]
                    chunk_data = chunk[1][0]

                    if node == "generate" and prev_node == "execute":
                        # st.session_state.user_interupt_message:
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": result}
                        )
                        st.rerun()

                    if node == "generate" and hasattr(chunk_data, "content"):
                        result += chunk_data.content
                        # Format and display streaming output
                        formatted_result = format_agent_output_for_display(result)
                        message_placeholder.markdown(formatted_result)
                    elif node == "execute" and hasattr(chunk_data, "content"):
                        # Handle case where content might be a list
                        content = chunk_data.content
                        if isinstance(content, list):
                            content = "".join(
                                str(item) for item in content if item["type"] == "text"
                            )
                        result += content
                        # Format and display streaming output
                        formatted_result = format_agent_output_for_display(result)
                        message_placeholder.markdown(formatted_result)
                    prev_node = node
        # Get generated files (images created during this step)
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"]
        all_images = []
        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(st.session_state.work_dir, ext)))

        # Filter to new files (created after this step started)
        # new_files = [f for f in all_images if f not in get_all_previous_files(step_num)]
        new_files = []
        # Extract solution content (clean results without execution details)
        solution_match = re.search(r"<solution>(.*?)</solution>", result, re.DOTALL)
        if solution_match:
            solution_content = solution_match.group(1).strip()

            # AGGRESSIVE CLEANING: Remove all execution artifacts from solution
            # (same cleaning logic as execute_single_step)
            solution_content = re.sub(
                r"<execute>.*?</execute>", "", solution_content, flags=re.DOTALL
            )
            solution_content = re.sub(
                r"<observation>.*?</observation>",
                "",
                solution_content,
                flags=re.DOTALL,
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
            solution_content = re.sub(r"Plan Update:.*?\n", "", solution_content)
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
            solution_content = re.sub(
                r"^---+$", "", solution_content, flags=re.MULTILINE
            )
            solution_content = re.sub(r"\n{3,}", "\n\n", solution_content)
            solution_content = solution_content.strip()

            if not solution_content or len(solution_content) < 20:
                solution_content = "✅ Analysis completed successfully.\n\nPlease see 'View Analysis Process' below for detailed execution steps and 'Figures' section for generated visualizations."
        else:
            observations = re.findall(
                r"<observation>(.*?)</observation>", result, re.DOTALL
            )
            solution_content = (
                observations[-1].strip()
                if observations
                else "Analysis completed. See process details below."
            )


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


def add_chat_message(role, content, files=None, timestamp=None):
    """Add a message to the chat history"""
    if timestamp is None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")

    message = {"role": role, "content": content, "timestamp": timestamp}

    if files:
        message["files"] = files

    st.session_state.chat_history.append(message)

    # Also update message_history for agent conversation continuity
    if "message_history" not in st.session_state:
        st.session_state.message_history = []
    st.session_state.message_history.append({"role": role, "content": content})


def build_agent_input_from_history(initial_prompt=None, include_initial=True):
    """Build agent input from conversation history

    Args:
        initial_prompt: The initial system prompt (only used on first run)
        include_initial: If True and initial_prompt provided, prepend it to history
    """
    agent_input = []

    # Start with initial prompt if this is the first run
    if include_initial and initial_prompt:
        agent_input.append(HumanMessage(content=initial_prompt))

    # Add conversation history from chat_history
    # This includes all user and assistant messages in chronological order
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            agent_input.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            agent_input.append(AIMessage(content=message["content"]))

    return agent_input


def display_chat_files(files):
    """Display files in chat message"""
    if not files:
        return

    st.markdown("**📎 첨부 파일:**")
    for file_path in files:
        filename = os.path.basename(file_path)
        col1, col2 = st.columns([3, 1])
        with col1:
            # Show image if it's an image file
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                st.image(file_path, use_container_width=True, caption=filename)
            else:
                st.markdown(f"📄 {filename}")
        with col2:
            try:
                with open(file_path, "rb") as f:
                    file_data = f.read()
                st.download_button(
                    label="⬇️",
                    data=file_data,
                    file_name=filename,
                    mime="application/octet-stream",
                    key=f"chat_download_{filename}_{len(st.session_state.chat_history)}",
                )
            except Exception as e:
                st.error(f"파일 로드 실패: {filename}")


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


# Main page logo
if LOGO_COLOR_BASE64 and LOGO_MONO_BASE64:
    st.markdown(
        f"""
    <div style="text-align: center; margin-bottom: 1rem; line-height: 0;">
        <img src="data:image/svg+xml;base64,{LOGO_COLOR_BASE64}" 
             class="logo-light main-logo" alt="OMICS-HORIZON Logo" 
             style="max-width: 600px; height: auto; margin: 0 auto;">
        <img src="data:image/svg+xml;base64,{LOGO_MONO_BASE64}" 
             class="logo-dark main-logo" alt="OMICS-HORIZON Logo" 
             style="max-width: 600px; height: auto; margin: 0 auto;">
    </div>
    """,
        unsafe_allow_html=True,
    )

# Title section
# st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem; margin-top: 0.5rem; margin-bottom: 2rem;">AI-Powered Transcriptomic Analysis Platform</p>', unsafe_allow_html=True)
st.markdown("---")

# Top row: 2 panels side by side
col1, col2 = st.columns(2)

# Panel 1: Data Upload & Briefing (Left)
with col1:
    st.markdown(
        f'<div class="panel-header">{t("panel1_title")}</div>', unsafe_allow_html=True
    )

    # File uploader
    uploaded_data = st.file_uploader(
        t("upload_data"),
        type=["csv", "xlsx", "xls", "tsv", "txt", "json", "gz"],
        accept_multiple_files=True,
        key="data_uploader",
    )

    if uploaded_data:
        # Show uploaded files
        st.info(f"📁 Uploaded {len(uploaded_data)} file(s)")
        for file in uploaded_data:
            st.text(f"  • {file.name}")

        # Analyze button
        if st.button(t("analyze_data"), type="primary", key="analyze_data"):
            # Save files
            file_names = []
            file_paths = []
            for file in uploaded_data:
                file_name = save_uploaded_file(file)
                file_names.append(file_name)
                file_paths.append(os.path.join(st.session_state.work_dir, file_name))
                if file_name not in st.session_state.data_files:
                    st.session_state.data_files.append(file_name)

            # Direct LLM analysis without agent (no history needed)
            with st.spinner("📊 Analyzing data files..."):
                result = analyze_data_direct(file_paths)
            st.session_state.data_briefing = result
            st.rerun()

    # Display briefing
    if st.session_state.data_briefing:
        st.markdown("---")
        st.markdown(f"### {t('data_briefing')}")
        st.markdown(st.session_state.data_briefing)


# Panel 2: Paper Upload & Method Extraction (Right)
with col2:
    st.markdown(
        f'<div class="panel-header">{t("panel2_title")}</div>', unsafe_allow_html=True
    )

    # File uploader
    uploaded_paper = st.file_uploader(
        t("upload_paper"),
        type=["pdf", "txt", "doc", "docx"],
        accept_multiple_files=False,
        key="paper_uploader",
    )

    if uploaded_paper:
        st.info(f"📄 Uploaded: {uploaded_paper.name}")

        # Extraction mode selector
        extraction_mode = st.radio(
            "추출 방식:" if st.session_state.language == "ko" else "Extraction Mode:",
            options=[
                (
                    "integrated",
                    (
                        "🎯 Results + Methods (추천)"
                        if st.session_state.language == "ko"
                        else "🎯 Results + Methods (Recommended)"
                    ),
                ),
                (
                    "methods_only",
                    (
                        "📋 Methods만"
                        if st.session_state.language == "ko"
                        else "📋 Methods Only"
                    ),
                ),
                (
                    "results_only",
                    (
                        "📊 Results만"
                        if st.session_state.language == "ko"
                        else "📊 Results Only"
                    ),
                ),
            ],
            format_func=lambda x: x[1],
            horizontal=True,
            key="extraction_mode",
            help=(
                "• Results+Methods: 분석 순서와 세부 방법을 통합 추출\n• Methods만: 기존 방식\n• Results만: 분석 순서만 추출"
                if st.session_state.language == "ko"
                else "• Results+Methods: Extract analysis order and detailed methods\n• Methods only: Traditional approach\n• Results only: Extract analysis sequence only"
            ),
        )

        mode = extraction_mode[0]

        # Extract method button
        if st.button(t("extract_workflow"), type="primary", key="extract_method"):
            # Save file
            file_name = save_uploaded_file(uploaded_paper)
            if file_name not in st.session_state.paper_files:
                st.session_state.paper_files.append(file_name)

            spinner_text = {
                "integrated": "📖 Extracting workflow by analyzing Results and Methods sections...",
                "methods_only": "📖 Extracting workflow from Methods section...",
                "results_only": "📖 Extracting analysis sequence from Results section...",
            }

            with st.spinner(spinner_text[mode]):
                # Extract workflow with selected mode
                result = extract_workflow_from_paper(
                    os.path.join(st.session_state.work_dir, file_name), mode=mode
                )
                st.session_state.analysis_method = result
            st.success(f"✅ Workflow extraction complete! ({extraction_mode[1]})")
            st.rerun()
    # Display and edit method
    if st.session_state.analysis_method:
        st.markdown("---")

        # No need for complex extraction - result is already clean
        clean_method = st.session_state.analysis_method

        # Show method in tabs
        method_tab1, method_tab2 = st.tabs(["📋 Analysis Workflow", "✏️ Edit"])

        with method_tab1:
            st.markdown("**🔬 Extracted Analysis Steps**")

            # Display as simple numbered list
            if clean_method:
                st.markdown(clean_method)
            else:
                st.warning("No method extracted. Please edit to add steps.")

        with method_tab2:
            st.info("💡 Format: Numbered list with tool names and parameters")

            edited_method = st.text_area(
                "Analysis Steps",
                value=clean_method,
                height=500,
                key="method_editor",
                placeholder="1. Preprocessing: log2 transformation using tool X\n2. DEG analysis: DESeq2 with |log2FC| > 2, p < 0.01\n3. Clustering: hierarchical clustering, heatmap\n...",
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "💾 Save",
                    key="save_method",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.analysis_method = edited_method
                    st.success("✅ Saved!")
                    st.rerun()

            with col2:
                if st.button("🔄 Reset", key="reset_method", use_container_width=True):
                    st.rerun()

    elif st.button("✍️ Write Custom Method", key="write_custom"):
        st.session_state.analysis_method = """1. Preprocessing: describe preprocessing, mention tools
2. Quality control: filtering criteria
3. Statistical analysis: test name, parameters (e.g., p < 0.05)
4. Clustering: method, visualization
5. Enrichment analysis: tool name, database
..."""
        st.rerun()

# Check if ready to start
if st.session_state.data_files and st.session_state.analysis_method:
    # Parse analysis steps
    analysis_steps = parse_analysis_steps(st.session_state.analysis_method)

    if analysis_steps:
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
        st.info(
            """
        💡 **Interactive Mode**: Execute analysis step-by-step with full control.
        - ▶️ Start each step when ready
        - 💬 Provide feedback to refine results
        - 🔄 Re-run steps as needed
        - Each step builds on previous results
        """
        )

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
                        "🙋 당신:" if st.session_state.language == "ko" else "🙋 You:"
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
                    "🚀 Ask" if st.session_state.language == "en" else "🚀 질문"
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
                    st.caption("• 이 p-value는 무엇을 나타내나요?")
                else:
                    st.caption("**Example questions:**")
                    st.caption("• What were the main findings in Step 2?")
                    st.caption("• Why was this statistical test chosen?")
                    st.caption("• Can you explain the volcano plot?")
                    st.caption("• What do these p-values indicate?")
    else:
        st.info(t("qa_no_analysis"))

    st.markdown("---")

    # Analysis Refinement Section (only show if workflow is completed)
    if is_workflow_completed():
        st.markdown(f"## {t('refinement_title')}")

        with st.expander(t("refinement_expander"), expanded=False):
            st.markdown(f"**{t('refinement_desc')}**")
            st.caption(t("refinement_examples"))

            # Refinement input
            refinement_request = st.text_area(
                t("refinement_placeholder"),
                key="refinement_request",
                height=100,
                placeholder=t("refinement_example"),
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    t("refinement_plan_button"),
                    key="get_refinement_plan",
                    use_container_width=True,
                ):
                    if refinement_request and refinement_request.strip():
                        with st.spinner("🤔 Analyzing your refinement request..."):
                            plan = apply_analysis_refinement(refinement_request.strip())
                        st.markdown(f"### {t('refinement_plan_title')}")
                        st.info(plan)
                    else:
                        st.warning("Please describe what refinement you want to make.")

            with col2:
                # Step selection for specific refinement
                if st.session_state.analysis_method:
                    try:
                        analysis_steps = parse_analysis_steps(
                            st.session_state.analysis_method
                        )
                        step_options = [
                            f"Step {s['step_num']}: {s['title'][:30]}..."
                            for s in analysis_steps
                            if st.session_state.steps_state.get(s["step_num"], {}).get(
                                "status"
                            )
                            == "completed"
                        ]
                    except:
                        step_options = []

                    if step_options:
                        selected_step_display = st.selectbox(
                            t("refinement_target_step"),
                            ["All steps"] + step_options,
                            key="refinement_target_step",
                        )

                        if st.button(
                            t("refinement_apply_button"),
                            key="apply_refinement",
                            use_container_width=True,
                            type="primary",
                        ):
                            if refinement_request and refinement_request.strip():
                                # Parse target step
                                target_step = None
                                if selected_step_display != "All steps":
                                    target_step = int(
                                        selected_step_display.split(":")[0].replace(
                                            "Step ", ""
                                        )
                                    )

                                with st.spinner("🔄 Applying refinement..."):
                                    if target_step:
                                        result = execute_refinement_action(
                                            refinement_request.strip(), target_step
                                        )
                                    else:
                                        # General refinement - get plan first
                                        result = apply_analysis_refinement(
                                            refinement_request.strip()
                                        )

                                if "✅" in result:
                                    st.success(result)
                                    st.rerun()  # Refresh to show updated results
                                else:
                                    st.error(result)
                            else:
                                st.warning(
                                    "Please describe what refinement you want to apply."
                                )
                    else:
                        st.info("No completed steps available for refinement.")

        st.markdown("---")

    # Control Panel and Session Info
    st.markdown(f"## {t('control_panel')}")

    st.markdown(f"### {t('session_info')}")
    st.info(
        f"""
    - Data files: {len(st.session_state.data_files)}
    - Paper files: {len(st.session_state.paper_files)}
    - Method defined: {'✅' if st.session_state.analysis_method else '❌'}
    - Work directory: `{st.session_state.work_dir.lstrip('/workdir_efs/jhjeon/Biomni/streamlit_workspace/')}`
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
           
        ### {t('refinement_instructions_title')}
        - {t('refinement_instructions_1')}
        - {t('refinement_instructions_2')}
        - {t('refinement_instructions_3')}
        - {t('refinement_instructions_4')}

        ### Tips:
        - Each step builds on previous results
        - You can re-run any step with different parameters
        - Use natural language for feedback (e.g., "Use p-value < 0.01")
        - After completion, use refinement tools for fine-tuning
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
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # When run directly (not from LIMS)
    run_omicshorizon_app(from_lims=False, workspace_path=None)
