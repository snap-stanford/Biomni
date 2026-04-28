"""
Gradio UI Layout Configuration for CAi Agent.

This module contains all UI styling, theme configuration, and layout building logic.
Simplified version focusing on functionality over complex styling.
"""

import gradio as gr

# ========== Theme Configuration ==========

# ========== Theme Configuration ==========

THEME = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg,
)

# ========== CSS Styles ==========

CSS = """
/* Global Container */
.gradio-container {
    max-width: 100% !important;
    padding: 12px 16px !important;
}

/* Top Navigation Bar */
.topbar {
    display: flex;
    align-items: center;
    padding: 12px 16px;
    background: var(--block-background-fill);
    border: 1px solid var(--block-border-color);
    border-radius: 12px;
    margin-bottom: 12px;
    gap: 16px;
    box-shadow: var(--shadow-drop);
}

.topbar-title {
    font-size: 1.3rem;
    font-weight: 700;
    flex: 1;
    color: var(--body-text-color);
}

.topbar-sub {
    font-size: 0.8rem;
    color: var(--body-text-color-subdued);
    background: var(--background-fill-secondary);
    padding: 2px 8px;
    border-radius: 12px;
}

/* Sidebar & Accordion */
.sidebar {
    padding: 4px !important;
}

.sidebar-accordion {
    background: var(--block-background-fill) !important;
    border: 1px solid var(--block-border-color) !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* Input Card */
.input-card {
    border-radius: 12px;
    background: var(--block-background-fill);
    border: 1px solid var(--block-border-color);
    padding: 12px 16px 8px 16px;
    margin-top: 8px;
    box-shadow: var(--shadow-drop);
}

/* Reference File Row */
.ref-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding-bottom: 8px;
    margin-bottom: 4px;
    border-bottom: 1px dashed var(--border-color-primary);
}

.ref-label {
    font-size: 0.75rem;
    color: var(--body-text-color-subdued);
    font-weight: 600;
    white-space: nowrap;
}

/* File Table */
.file-table {
    margin-bottom: 8px !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 8px;
    overflow: hidden;
}

.file-table tbody tr {
    height: 32px;
    font-size: 0.85rem;
    cursor: pointer;
    border-bottom: 1px solid var(--border-color-secondary);
}

.file-table tbody tr:hover {
    background: var(--background-fill-secondary);
}

/* 重新设计表头：取消突兀的亮蓝色，改为专业低调的浅灰/深灰 */
.file-table thead th {
    background: var(--background-fill-secondary) !important;
    color: var(--body-text-color) !important;
    font-weight: 600;
    font-size: 0.75rem;
    border-bottom: 1px solid var(--border-color-primary) !important;
}

/* UI Elements */
button {
    border-radius: 8px !important;
    font-weight: 600 !important;
}

.verify-card {
    max-width: 420px;
    margin: 100px auto;
    padding: 40px;
    border-radius: 16px;
    background: var(--block-background-fill);
    border: 1px solid var(--block-border-color);
    box-shadow: var(--shadow-drop-lg);
}

.accordion {
    border-radius: 8px !important;
}

.fade-in {
    animation: fadeIn 0.3s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
"""


# ========== Constants ==========

CHAT_HEIGHT = 640


# ========== Layout Builders ==========


def _build_verification_page(ui_instance):
    """Build the access code verification page."""
    with gr.Column(elem_classes="verify-card fade-in"):
        gr.Markdown(
            """
            <div style="text-align: center;">
                <h1 style="color: #667eea; font-size: 2.5rem; margin-bottom: 0.5rem;">
                    CAi Agent
                </h1>
                <p style="color: #6b7280; font-size: 0.95rem;">请输入访问码以继续使用</p>
            </div>
            """
            # 🤖 CAi Agent
        )
        access_code_input = gr.Textbox(
            label="Access Code",
            type="password",
            placeholder="输入访问码...",
            container=True,
        )
        access_error_msg = gr.Markdown(visible=False)
        verify_btn = gr.Button("🔓 验证身份", variant="primary", size="lg")

    return access_code_input, access_error_msg, verify_btn


def _build_topbar():
    """Build the top navigation bar."""
    with gr.Row(elem_classes="topbar fade-in"):
        gr.HTML(
            """
            <div style="display: flex; align-items: center; gap: 12px; flex: 1;">
                <span class="topbar-title">🤖 CAi Copilot</span>
                <span class="topbar-sub">⚡ Powered by AILAB</span>
            </div>
            """
        )
        with gr.Row(scale=0):
            export_history_btn = gr.Button("💾 导出会话", variant="primary", size="sm")
            clear_files_btn = gr.Button("🗑️ 清空文件", variant="secondary", size="sm")
            clear_all_btn = gr.Button("🔄 重置全部", variant="stop", size="sm")

    return export_history_btn, clear_files_btn, clear_all_btn


def _build_sidebar(ui_instance):
    """构建可折叠的文件管理侧边栏"""
    # 将 scale 设置为 1，但允许内部内容折叠
    with gr.Column(scale=1, min_width=300, elem_classes="sidebar fade-in"):
        # 👇 使用 gr.Accordion 包裹整个侧边栏内容
        with gr.Accordion("📁 工作区管理", open=True, elem_classes="sidebar-accordion"):
            file_viewer = gr.Dataframe(
                headers=["文件名"],
                value=ui_instance._get_current_files(),
                interactive=False,
                elem_classes="file-table",
                wrap=True,
                row_count=4,  # 👇 进一步缩短显示行数，让界面更紧凑
            )

            # 内部原有的预览折叠保持不变
            with gr.Accordion("👁️ 内容预览", open=False, elem_classes="accordion"):
                preview_image = gr.Image(interactive=False, visible=False, show_label=False)
                preview_code = gr.Code(interactive=False, visible=False, show_label=False)
                preview_pdf = gr.HTML(visible=False)
                preview_fallback = gr.File(label="📥 下载文件", interactive=False, visible=False)

    return file_viewer, preview_image, preview_code, preview_pdf, preview_fallback


def _build_chat_area():
    """Build the dual chatbot area."""
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div style="text-align: center; margin-bottom: 8px;">
                    <span style="font-size: 1rem; font-weight: 600; color: var(--color-accent);">
                        💬 主对话
                    </span>
                </div>
                """
            )
            main_chatbot = gr.Chatbot(
                label="",
                type="messages",
                scale=1,
                min_height=500,
                show_copy_button=True,
                elem_classes="chatbot-main fade-in",
                avatar_images=(
                    None,
                    "https://api.dicebear.com/7.x/bottts/svg?seed=cai&backgroundColor=4f46e5",
                ),
            )

        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div style="text-align: center; margin-bottom: 8px;">
                    <span style="font-size: 1rem; font-weight: 600; color: var(--body-text-color-subdued);">
                        ⚙️ 执行日志
                    </span>
                </div>
                """
            )
            innerloop_chatbot = gr.Chatbot(
                label="",
                type="messages",
                scale=1,
                min_height=500,
                show_copy_button=True,
                elem_classes="chatbot-inner fade-in",
            )

    return main_chatbot, innerloop_chatbot


def _build_input_area(ui_instance):
    """Build the input area with file reference and text input."""
    with gr.Column(elem_classes="input-card fade-in"):
        with gr.Row(elem_classes="ref-row"):
            gr.HTML('<span class="ref-label">📎 引用文件</span>')
            ref_dropdown = gr.Dropdown(
                choices=ui_instance._get_dropdown_choices(),
                multiselect=True,
                show_label=False,
                container=False,
                interactive=True,
                scale=1,
            )
        prompt_input = gr.MultimodalTextbox(
            interactive=True,
            file_count="multiple",
            placeholder="💭 输入指令，或拖入文件上传...",
            show_label=False,
            container=False,
        )

    return ref_dropdown, prompt_input


def _setup_event_handlers(
    ui_instance,
    file_viewer,
    preview_image,
    preview_code,
    preview_pdf,
    preview_fallback,
    prompt_input,
    ref_dropdown,
    innerloop_chatbot,
    main_chatbot,
    export_history_btn,
    clear_files_btn,
    clear_all_btn,
):
    """Setup all event handlers for the UI."""
    # File preview
    file_viewer.select(
        fn=ui_instance._preview_selected_file,
        inputs=[],
        outputs=[preview_image, preview_code, preview_pdf, preview_fallback],
    )

    # Message submission
    prompt_input.submit(
        fn=ui_instance.generate_response,
        inputs=[prompt_input, ref_dropdown, innerloop_chatbot, main_chatbot],
        outputs=[innerloop_chatbot, main_chatbot, file_viewer, ref_dropdown],
        show_progress="full",
    ).then(lambda: ({"text": "", "files": []}, []), None, [prompt_input, ref_dropdown])

    # Helper function to hide all previews
    hide_previews = lambda: (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )

    # Export conversation history to PDF
    export_history_btn.click(
        fn=ui_instance._save_conversation_history,
        outputs=[file_viewer, ref_dropdown],
    )

    # Clear files only
    clear_files_btn.click(
        fn=ui_instance._clear_workspace,
        outputs=[file_viewer, ref_dropdown],
    ).then(
        hide_previews,
        inputs=None,
        outputs=[preview_image, preview_code, preview_pdf, preview_fallback],
    )

    # Clear files and chat
    clear_all_btn.click(
        fn=ui_instance._clear_all,
        outputs=[innerloop_chatbot, main_chatbot, file_viewer, ref_dropdown],
    ).then(
        hide_previews,
        inputs=None,
        outputs=[preview_image, preview_code, preview_pdf, preview_fallback],
    )


# ========== Main Layout Builder ==========


def build_layout(ui_instance):
    """
    Build the complete Gradio UI layout.

    Args:
        ui_instance: Instance of AgentGradioUI

    Returns:
        gr.Blocks: Configured Gradio interface
    """
    with gr.Blocks(theme=THEME, title="CAi Agent", css=CSS, fill_width=True, fill_height=True) as demo:
        verification_container = gr.Group(visible=ui_instance.require_verification)
        main_interface_container = gr.Group(visible=not ui_instance.require_verification)

        # Verification Page
        with verification_container:
            access_code_input, access_error_msg, verify_btn = _build_verification_page(ui_instance)

            verify_btn.click(
                fn=ui_instance._verify_access_code,
                inputs=[access_code_input],
                outputs=[verification_container, main_interface_container, access_error_msg],
            )

        # Main Interface
        with main_interface_container:
            # Top bar
            export_history_btn, clear_files_btn, clear_all_btn = _build_topbar()

            with gr.Row(equal_height=False):
                # Sidebar
                file_viewer, preview_image, preview_code, preview_pdf, preview_fallback = _build_sidebar(ui_instance)

                # 2. 👇 把这里的主区域权重加大！从 3 改成 4 或 5
                with gr.Column(scale=5):
                    main_chatbot, innerloop_chatbot = _build_chat_area()
                    ref_dropdown, prompt_input = _build_input_area(ui_instance)

        # Setup event handlers
        _setup_event_handlers(
            ui_instance,
            file_viewer,
            preview_image,
            preview_code,
            preview_pdf,
            preview_fallback,
            prompt_input,
            ref_dropdown,
            innerloop_chatbot,
            main_chatbot,
            export_history_btn,
            clear_files_btn,
            clear_all_btn,
        )

    return demo
