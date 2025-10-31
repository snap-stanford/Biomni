"\"\"\"Control panel rendering helpers for Omics Horizon UI.\"\"\""

from __future__ import annotations

import streamlit as st


def render_control_panel(t, llm_model: str, workspace_display_fn) -> None:
    """Render the right-hand control panel with session utilities."""
    st.markdown(f"## {t('control_panel')}")

    st.markdown(f"### {t('session_info')}")
    workspace_display = workspace_display_fn(st.session_state.work_dir)
    st.info(
        f"""
    - Data files: {len(st.session_state.data_files)}
    - Paper files: {len(st.session_state.paper_files)}
    - Method defined: {'✅' if st.session_state.analysis_method else '❌'}
    - Work directory: `{workspace_display}`
    """
    )

    st.markdown("---")

    if st.session_state.steps_state:
        if st.button("🔄 Reset Analysis", key="reset_analysis", use_container_width=True):
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

    with st.expander(t("instructions")):
        st.markdown(
            f"""
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
    st.text(f"Model: {llm_model}")

    st.markdown("---")

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
