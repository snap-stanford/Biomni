"\"\"\"Static resources (translations, CSS, logos) for Omics Horizon.\"\"\""

from __future__ import annotations

import base64
from typing import Dict

import streamlit as st

from .config import LOGO_COLOR_PATH, LOGO_MONO_PATH


def load_logo_base64(logo_path: str) -> str:
    """Load and cache logo image as base64-encoded string."""
    if not logo_path:
        return ""
    try:
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        st.warning(f"Logo file not found: {logo_path}")
        return ""


def load_logo_assets() -> tuple[str, str]:
    """Convenience wrapper to load both logo variants."""
    return load_logo_base64(LOGO_COLOR_PATH), load_logo_base64(LOGO_MONO_PATH)


GLOBAL_CSS_TEMPLATE = """
<style>
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }}
    .panel-header {{
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin-bottom: 1rem;
    }}
    .stTextArea textarea {{
        font-family: monospace;
    }}
    div[data-testid="stExpander"] {{
        border: 2px solid #e0e0e0;
        border-radius: 10px;
    }}

    {{sidebar_width_rule}}

    .logo-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1rem 0;
        margin-bottom: -2rem;
    }}
    .logo-container img {{
        max-width: 100%;
        height: auto;
    }}

    .logo-light {{
        display: block !important;
    }}
    .logo-dark {{
        display: none !important;
    }}

    @media (prefers-color-scheme: dark) {{
        .logo-light {{
            display: none !important;
        }}
        .logo-dark {{
            display: block !important;
        }}
    }}

    [data-theme="dark"] .logo-light,
    [data-baseweb-theme="dark"] .logo-light,
    .stApp[data-theme="dark"] .logo-light {{
        display: none !important;
    }}

    [data-theme="dark"] .logo-dark,
    [data-baseweb-theme="dark"] .logo-dark,
    .stApp[data-theme="dark"] .logo-dark {{
        display: block !important;
    }}

    .main-logo {{
        margin: 0 auto;
        position: relative;
    }}

    .logo-light {{
        z-index: 2;
    }}
    .logo-dark {{
        z-index: 1;
    }}
</style>
"""


TRANSLATIONS: Dict[str, Dict[str, str]] = {
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
        "qa_no_analysis": "💡 Start the analysis chat to ask follow-up questions",
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
        "qa_no_analysis": "💡 분석 대화를 시작하면 질문할 수 있어요",
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
