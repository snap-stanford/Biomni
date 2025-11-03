"\"\"\"Primary data/method panels for Omics Horizon UI.\"\"\""

from __future__ import annotations

import os
from typing import Callable

import streamlit as st


SaveFileFn = Callable[[any], str]
AnalyzeDataFn = Callable[[list[str]], str]
ExtractWorkflowFn = Callable[[str, str], str]


def _render_lims_data_panel(t) -> None:
    """Render read-only data summary when launched from LIMS."""
    st.markdown(
        '<div class="panel-header">📊 Data from LIMS</div>',
        unsafe_allow_html=True,
    )

    has_any_content = False
    md_parts: list[str] = []

    if "selected_data_files" in st.session_state and st.session_state.selected_data_files:
        has_any_content = True
        md_parts.append("**Selected files from LIMS:**")
        for file_info in st.session_state.selected_data_files:
            md_parts.append(f"- {file_info['name']} ({file_info['extension']})")
        md_parts.append("")

    if st.session_state.data_files:
        has_any_content = True
        md_parts.append(
            f"✅ Loaded {len(st.session_state.data_files)} file(s) from LIMS:"
        )
        for filename in st.session_state.data_files:
            md_parts.append(f"- {filename}")
        md_parts.append("")

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
            st.markdown(full_md)
    else:
        st.warning("⚠️ No data files loaded from LIMS")


def _render_local_data_panel(
    t, save_uploaded_file: SaveFileFn, analyze_data_direct: AnalyzeDataFn
) -> None:
    """Render upload controls when running standalone (not from LIMS)."""
    st.markdown(
        f'<div class="panel-header">{t("panel1_title")}</div>',
        unsafe_allow_html=True,
    )

    uploaded_data = st.file_uploader(
        t("upload_data"),
        type=["csv", "xlsx", "xls", "tsv", "txt", "json", "gz"],
        accept_multiple_files=True,
        key="data_uploader",
    )

    if uploaded_data:
        st.info(f"📁 Uploaded {len(uploaded_data)} file(s)")
        for file in uploaded_data:
            st.text(f"  • {file.name}")

        if st.button(t("analyze_data"), type="primary", key="analyze_data"):
            file_paths: list[str] = []
            for file in uploaded_data:
                file_name = save_uploaded_file(file)
                if file_name not in st.session_state.data_files:
                    st.session_state.data_files.append(file_name)
                file_paths.append(os.path.join(st.session_state.work_dir, file_name))

            with st.spinner("📊 Analyzing data files..."):
                result = analyze_data_direct(file_paths)
            st.session_state.data_briefing = result
            st.rerun()

    if st.session_state.data_briefing:
        st.markdown("---")
        st.markdown(f"### {t('data_briefing')}")
        st.markdown(st.session_state.data_briefing)


def _render_paper_panel(
    t,
    save_uploaded_file: SaveFileFn,
    extract_workflow_from_paper: ExtractWorkflowFn,
) -> None:
    st.markdown(
        f'<div class="panel-header">{t("panel2_title")}</div>',
        unsafe_allow_html=True,
    )

    uploaded_paper = st.file_uploader(
        t("upload_paper"),
        type=["pdf", "txt", "doc", "docx"],
        accept_multiple_files=False,
        key="paper_uploader",
    )

    if uploaded_paper:
        st.info(f"📄 Uploaded: {uploaded_paper.name}")
        extraction_mode = st.radio(
            "추출 방식:" if st.session_state.language == "ko" else "Extraction Mode:",
            options=[
                (
                    "integrated",
                    "🎯 Results + Methods (추천)"
                    if st.session_state.language == "ko"
                    else "🎯 Results + Methods (Recommended)",
                ),
                (
                    "methods_only",
                    "📋 Methods만"
                    if st.session_state.language == "ko"
                    else "📋 Methods Only",
                ),
                (
                    "results_only",
                    "📊 Results만"
                    if st.session_state.language == "ko"
                    else "📊 Results Only",
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

        if st.button(t("extract_workflow"), type="primary", key="extract_method"):
            file_name = save_uploaded_file(uploaded_paper)
            if file_name not in st.session_state.paper_files:
                st.session_state.paper_files.append(file_name)

            spinner_text = {
                "integrated": "📖 Extracting workflow by analyzing Results and Methods sections...",
                "methods_only": "📖 Extracting workflow from Methods section...",
                "results_only": "📖 Extracting analysis sequence from Results section...",
            }
            with st.spinner(spinner_text[mode]):
                result = extract_workflow_from_paper(
                    os.path.join(st.session_state.work_dir, file_name), mode=mode
                )
                st.session_state.analysis_method = result
                # Initialize editor content with extracted result
                st.session_state.method_editor_content = result
            st.success(f"✅ Workflow extraction complete! ({extraction_mode[1]})")
            st.rerun()

    uploaded_md_parts: list[str] = []
    if "paper_files" in st.session_state and st.session_state.paper_files:
        uploaded_md_parts.append(
            f"**Uploaded Paper Files:** ({len(st.session_state.paper_files)} file(s))"
        )
        for filename in st.session_state.paper_files:
            uploaded_md_parts.append(f"- {filename}")
        uploaded_md_parts.append("")

    if not st.session_state.analysis_method and uploaded_md_parts:
        full_md = "\n".join(uploaded_md_parts)
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

    if st.session_state.analysis_method:
        clean_method = st.session_state.analysis_method
        method_tab1, method_tab2 = st.tabs(["📋 Analysis Workflow", "✏️ Edit"])

        with method_tab1:
            method_md_parts = []
            if uploaded_md_parts:
                method_md_parts.extend(uploaded_md_parts)
            method_md_parts.append("### 🔬 Extracted Analysis Steps")
            method_md_parts.append("")
            method_md_parts.append(
                clean_method
                if clean_method
                else "*No method extracted. Please edit to add steps.*"
            )

            full_md = "\n".join(method_md_parts)
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

        with method_tab2:
            st.info("💡 Format: Numbered list with tool names and parameters")
            # Use session state to preserve edited content, but initialize with extracted method
            if "method_editor_content" not in st.session_state:
                st.session_state.method_editor_content = clean_method
            elif not st.session_state.method_editor_content and clean_method:
                # If editor is empty but we have extracted content, use extracted content
                st.session_state.method_editor_content = clean_method
            
            edited_method = st.text_area(
                "Analysis Steps",
                value=st.session_state.method_editor_content,
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
                    st.session_state.method_editor_content = edited_method
                    st.success("✅ Saved!")
                    st.rerun()

            with col2:
                if st.button("🔄 Reset", key="reset_method", use_container_width=True):
                    # Reset to the extracted method
                    st.session_state.method_editor_content = clean_method
                    st.rerun()
    elif st.button("✍️ Write Custom Method", key="write_custom"):
        st.session_state.analysis_method = """1. Statistical analysis: t-test, p-value < 0.1, |log2FC| > 2, Volcano plot.

2. Clustering: Heatmap with dendrogram

3. Enrichment analysis: GO enrichment analysis"""
        st.rerun()


def render_primary_panels(
    from_lims: bool,
    t,
    save_uploaded_file: SaveFileFn,
    analyze_data_direct: AnalyzeDataFn,
    extract_workflow_from_paper: ExtractWorkflowFn,
) -> None:
    """Render top-level data upload and paper workflow panels."""
    col1, col2 = st.columns(2)
    with col1:
        if from_lims:
            _render_lims_data_panel(t)
        else:
            _render_local_data_panel(t, save_uploaded_file, analyze_data_direct)

    with col2:
        _render_paper_panel(t, save_uploaded_file, extract_workflow_from_paper)
