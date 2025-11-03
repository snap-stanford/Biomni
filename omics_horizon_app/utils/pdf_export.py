"""PDF export utilities for Omics Horizon analysis results."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

import streamlit as st


def _markdown_to_text(markdown_text: str) -> str:
    """Convert markdown to plain text for PDF."""
    # Remove markdown links but keep text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", markdown_text)
    # Remove bold/italic markers
    text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^\*]+)\*", r"\1", text)
    # Remove code blocks (keep content)
    text = re.sub(r"```[\w]*\n(.*?)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Clean up extra whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_figure_paths(content: str, workspace_dir: str) -> list[tuple[str, Path]]:
    """Extract figure paths from content (supports [[FIGURE::path]] tokens and HTML img tags)."""
    figures: list[tuple[str, Path]] = []
    
    # Extract [[FIGURE::path]] tokens
    figure_pattern = r"\[\[FIGURE::(.*?)\]\]"
    for match in re.finditer(figure_pattern, content):
        path_str = match.group(1).strip()
        try:
            fig_path = Path(path_str)
            if not fig_path.is_absolute():
                fig_path = Path(workspace_dir) / fig_path
            if fig_path.exists() and fig_path.is_file():
                figures.append((path_str, fig_path))
        except Exception:
            continue
    
    # Extract from HTML img tags with base64 data URIs (we'll skip these for now)
    # Extract from HTML img tags with file paths
    img_pattern = r'<img[^>]+src=["\'](?:data:image/[^;]+;base64,[^"\']+|([^"\']+))["\']'
    for match in re.finditer(img_pattern, content):
        path_str = match.group(1)
        if path_str and not path_str.startswith("data:"):
            try:
                img_path = Path(path_str)
                if not img_path.is_absolute():
                    img_path = Path(workspace_dir) / img_path
                if img_path.exists() and img_path.is_file():
                    figures.append((path_str, img_path))
            except Exception:
                continue
    
    return figures


def generate_pdf_report(
    chat_history: list[dict[str, Any]],
    analysis_method: str,
    data_briefing: str,
    workspace_dir: str,
    output_path: Path | str,
) -> Path:
    """Generate a PDF report from analysis results.
    
    Args:
        chat_history: List of chat messages with role and content
        analysis_method: The analysis workflow/method
        data_briefing: Data briefing text
        workspace_dir: Workspace directory path
        output_path: Path where PDF should be saved
        
    Returns:
        Path to generated PDF file
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab is required for PDF export. Install it with: pip install reportlab"
        )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create PDF document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )
    
    # Container for the 'Flowable' objects
    story = []
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#1f77b4"),
        spaceAfter=30,
        alignment=TA_CENTER,
    )
    
    # Heading styles
    heading1_style = ParagraphStyle(
        "CustomHeading1",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#333333"),
        spaceAfter=12,
        spaceBefore=20,
    )
    
    heading2_style = ParagraphStyle(
        "CustomHeading2",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#666666"),
        spaceAfter=10,
        spaceBefore=15,
    )
    
    # Add title
    story.append(Paragraph("OmicsHorizon Analysis Report", title_style))
    story.append(Spacer(1, 0.2 * inch))
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"<i>Generated: {timestamp}</i>", styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))
    
    # Add data briefing section
    if data_briefing:
        story.append(Paragraph("<b>Data Briefing</b>", heading1_style))
        briefing_text = _markdown_to_text(data_briefing)
        for para in briefing_text.split("\n\n"):
            if para.strip():
                story.append(Paragraph(para.strip().replace("\n", "<br/>"), styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))
    
    # Add analysis method section
    if analysis_method:
        story.append(Paragraph("<b>Analysis Workflow</b>", heading1_style))
        method_text = _markdown_to_text(analysis_method)
        for para in method_text.split("\n\n"):
            if para.strip():
                story.append(Paragraph(para.strip().replace("\n", "<br/>"), styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))
    
    # Add chat history/conversation section
    if chat_history:
        story.append(Paragraph("<b>Analysis Conversation</b>", heading1_style))
        
        for msg in chat_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if not content:
                continue
            
            # Role header
            role_style = heading2_style
            if role == "assistant":
                role_text = "<b>Assistant:</b>"
            elif role == "user":
                role_text = "<b>User:</b>"
            else:
                role_text = f"<b>{role.title()}:</b>"
            
            story.append(Paragraph(role_text, role_style))
            
            # Convert content to plain text (removing HTML)
            text_content = _markdown_to_text(content)
            
            # Check for figures in content
            figures = _extract_figure_paths(content, workspace_dir)
            
            # Add text paragraphs
            paragraphs = [p.strip() for p in text_content.split("\n\n") if p.strip()]
            for para in paragraphs:
                # Skip if it's just a figure reference
                if "[[FIGURE::" not in para:
                    para_html = para.replace("\n", "<br/>")
                    # Escape special characters for reportlab
                    para_html = para_html.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    # Restore intentional breaks
                    para_html = para_html.replace("&lt;br/&gt;", "<br/>")
                    story.append(Paragraph(para_html, styles["Normal"]))
            
            # Add figures
            for fig_label, fig_path in figures:
                try:
                    # Check if image is readable
                    if fig_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:
                        # Calculate available space on page (leave margins)
                        # A4 size in points: 595.27 x 841.89
                        max_width_points = A4[0] - 144  # A4 width minus left+right margins (72*2 = 144)
                        max_height_points = A4[1] - 250  # A4 height minus top+bottom margins and spacing
                        
                        # Get image dimensions using PIL
                        try:
                            from PIL import Image as PILImage
                            pil_img = PILImage.open(str(fig_path))
                            img_width_px, img_height_px = pil_img.size
                            
                            # Convert pixels to points (assuming 96 DPI)
                            # 1 inch = 72 points, 1 inch = 96 pixels at 96 DPI
                            img_width_pts = img_width_px * (72.0 / 96.0)
                            img_height_pts = img_height_px * (72.0 / 96.0)
                            
                            # Calculate scale to fit within available space
                            scale_w = max_width_points / img_width_pts if img_width_pts > 0 else 1.0
                            scale_h = max_height_points / img_height_pts if img_height_pts > 0 else 1.0
                            scale = min(scale_w, scale_h, 1.0)  # Don't upscale, only shrink if needed
                            
                            # Calculate final size in points
                            final_width = img_width_pts * scale
                            final_height = img_height_pts * scale
                            
                            # Double check we don't exceed limits
                            final_width = min(final_width, max_width_points)
                            final_height = min(final_height, max_height_points)
                            
                        except (ImportError, Exception):
                            # Fallback: use simple width constraint if PIL not available
                            final_width = min(4.5 * inch, max_width_points)
                            final_height = None
                        
                        # Create Image with proper sizing
                        if final_height and final_height > 0:
                            img = Image(str(fig_path), width=final_width, height=final_height)
                        else:
                            img = Image(str(fig_path), width=final_width)
                        
                        story.append(Spacer(1, 0.1 * inch))
                        story.append(img)
                        story.append(Paragraph(f"<i>Figure: {fig_path.name}</i>", styles["Normal"]))
                        story.append(Spacer(1, 0.2 * inch))
                except Exception as e:
                    # If image can't be loaded, just add a note
                    story.append(Paragraph(f"<i>[Figure not available: {fig_path.name}: {str(e)}]</i>", styles["Normal"]))
            
            # Handle files attached to message
            if msg.get("files"):
                story.append(Spacer(1, 0.1 * inch))
                file_text = "<b>Attached files:</b> " + ", ".join(msg.get("files", []))
                story.append(Paragraph(file_text, styles["Normal"]))
            
            story.append(Spacer(1, 0.2 * inch))
        
        story.append(PageBreak())
    
    # Build PDF
    doc.build(story)
    
    return output_path


def export_to_pdf_button() -> None:
    """Render a button to export analysis results to PDF."""
    if not REPORTLAB_AVAILABLE:
        st.warning(
            "PDF export requires reportlab. Install with: pip install reportlab"
        )
        return
    
    chat_history = st.session_state.get("chat_history", [])
    analysis_method = st.session_state.get("analysis_method", "")
    data_briefing = st.session_state.get("data_briefing", "")
    workspace_dir = st.session_state.get("work_dir", "")
    
    if not chat_history and not analysis_method:
        st.info("No analysis results to export yet.")
        return
    
    if st.button("📄 Export to PDF", key="export_pdf", use_container_width=True):
        try:
            with st.spinner("Generating PDF report..."):
                # Create output filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_filename = f"analysis_report_{timestamp}.pdf"
                pdf_path = Path(workspace_dir) / pdf_filename
                
                # Generate PDF
                output_path = generate_pdf_report(
                    chat_history=chat_history,
                    analysis_method=analysis_method,
                    data_briefing=data_briefing,
                    workspace_dir=workspace_dir,
                    output_path=pdf_path,
                )
                
                # Read PDF and provide download
                with open(output_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                    
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    key="download_pdf",
                    use_container_width=True,
                )
                
                st.success(f"✅ PDF generated: {pdf_filename}")
        except Exception as e:
            st.error(f"Failed to generate PDF: {str(e)}")

