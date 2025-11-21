"""
Workflow generation module for workflow recommendations.
This module contains methods for generating workflow recommendations from files and research papers.
"""

import os
import re
import base64
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from biomni.llm import get_llm


class WorkflowRecommendation(BaseModel):
    """Schema for workflow recommendation output."""

    workflow_title: str = Field(
        description="A clear, concise title for the workflow (max 10-15 words)"
    )
    key_steps: List[str] = Field(
        description="Provide exactly 3-5 essential, actionable steps. Each step should be concise (1-2 sentences max)",
    )


class WorkflowGenerator:
    """Class for generating workflow recommendations from various sources."""

    # File extension categories (shared across methods)
    TEXT_READABLE_EXTENSIONS = {
        ".txt",
        ".csv",
        ".tsv",
        ".json",
        ".xml",
        ".html",
        ".md",
        ".log",
        ".py",
        ".r",
        ".sh",
        ".yaml",
        ".yml",
        ".ini",
        ".cfg",
        ".conf",
    }
    EXCEL_EXTENSIONS = {".xlsx", ".xls"}
    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".svg",
        ".tiff",
        ".tif",
        ".webp",
    }
    MIME_TYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".svg": "image/svg+xml",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".webp": "image/webp",
    }

    def __init__(self, prompts_dir=None):
        """Initialize WorkflowGenerator.

        Args:
            prompts_dir: Directory containing prompt template files
        """
        self.prompts_dir = prompts_dir or os.path.join(
            os.path.dirname(__file__), "prompts"
        )

    def _load_prompt_template(self, filename):
        """Load a prompt template from file.

        Args:
            filename: Name of the prompt template file

        Returns:
            str: Prompt template content with format() capability
        """
        prompt_path = os.path.join(self.prompts_dir, filename)
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _format_file_size(file_size):
        """Format file size in human-readable format.

        Args:
            file_size: File size in bytes

        Returns:
            str: Formatted file size string
        """
        if file_size < 1024:
            return f"{file_size} bytes"
        elif file_size < 1024 * 1024:
            return f"{file_size / 1024:.2f} KB"
        else:
            return f"{file_size / (1024 * 1024):.2f} MB"

    @staticmethod
    def _format_workflow_output(workflow_output):
        """Format structured output into readable workflow string.

        Args:
            workflow_output: WorkflowRecommendation object

        Returns:
            str: Formatted workflow string
        """
        workflow_text = f"""# {workflow_output.workflow_title}

## Key Steps
"""
        if workflow_output.key_steps:
            for i, step in enumerate(workflow_output.key_steps, 1):
                workflow_text += f"{i}. {step}\n"
        else:
            workflow_text += "No specific steps provided.\n"

        return workflow_text

    def _process_image_file(self, file_path, file_name, file_ext, file_size):
        """Process image file and return file info and image data.

        Args:
            file_path: Path to image file
            file_name: Name of the file
            file_ext: File extension
            file_size: File size in bytes

        Returns:
            tuple: (file_info_str, image_data_dict or None)
        """
        size_str = self._format_file_size(file_size)

        try:
            from PIL import Image

            with Image.open(file_path) as img:
                img_width, img_height = img.size

            file_info = (
                f"- {file_name} ({size_str}, resolution: {img_width}x{img_height} pixels)\n"
                f"  Type: Image file\n"
                f"  Path: {file_path}"
            )

            # Read and encode image as base64
            with open(file_path, "rb") as img_file:
                img_data = img_file.read()
                base64_image = base64.b64encode(img_data).decode("utf-8")

            mime_type = self.MIME_TYPES.get(file_ext, "image/png")
            image_data = {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            }

            return file_info, image_data

        except Exception as e:
            file_info = (
                f"- {file_name} ({size_str})\n"
                f"  Type: Image file (Error reading: {str(e)})\n"
                f"  Path: {file_path}"
            )
            return file_info, None

    def _process_excel_file(self, file_path, file_name, file_ext, file_size):
        """Process Excel file and return file info.

        Args:
            file_path: Path to Excel file
            file_name: Name of the file
            file_ext: File extension
            file_size: File size in bytes

        Returns:
            str: File info string
        """
        size_str = self._format_file_size(file_size)

        try:
            import pandas as pd

            df = pd.read_excel(file_path, nrows=5)
            num_sheets = len(pd.ExcelFile(file_path).sheet_names)
            total_rows = len(pd.read_excel(file_path, nrows=None))
            num_cols = len(df.columns)

            columns_preview = ", ".join(df.columns[:10].tolist())
            if len(df.columns) > 10:
                columns_preview += f", ... ({len(df.columns) - 10} more columns)"

            content_preview = df.to_string(index=False, max_rows=5, max_cols=10)
            if len(content_preview) > 1000:
                content_preview = content_preview[:1000] + "\n... (truncated)"

            return (
                f"- {file_name} ({size_str})\n"
                f"  Type: Excel file ({file_ext})\n"
                f"  Path: {file_path}\n"
                f"  Sheets: {num_sheets}, Rows: {total_rows}, Columns: {num_cols}\n"
                f"  Column names: {columns_preview}\n"
                f"  Data preview (first 5 rows):\n```\n{content_preview}\n```"
            )

        except Exception as e:
            return (
                f"- {file_name} ({size_str})\n"
                f"  Type: Excel file (Error reading: {str(e)})\n"
                f"  Path: {file_path}"
            )

    def _process_text_file(self, file_path, file_name, file_ext, file_size):
        """Process text-readable file and return file info.

        Args:
            file_path: Path to text file
            file_name: Name of the file
            file_ext: File extension
            file_size: File size in bytes

        Returns:
            str: File info string
        """
        size_str = self._format_file_size(file_size)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = []
                char_count = 0
                max_lines = 5
                max_chars = 1000

                for i, line in enumerate(f):
                    if i >= max_lines or char_count >= max_chars:
                        lines.append("... (truncated)")
                        break
                    lines.append(line.rstrip())
                    char_count += len(line)

                content_preview = "\n".join(lines)

            return (
                f"- {file_name} ({size_str})\n"
                f"  Type: Text file ({file_ext})\n"
                f"  Path: {file_path}\n"
                f"  Content preview:\n```\n{content_preview}\n```"
            )

        except UnicodeDecodeError:
            return (
                f"- {file_name} ({size_str})\n"
                f"  Type: Binary file\n"
                f"  Path: {file_path}"
            )
        except Exception as e:
            return (
                f"- {file_name} ({size_str})\n"
                f"  Type: Text file (Error reading: {str(e)})\n"
                f"  Path: {file_path}"
            )

    def _analyze_files(self, file_paths):
        """Analyze files and return file information and image data.

        Args:
            file_paths: List of file paths or single file path

        Returns:
            tuple: (files_info list, images_data list)
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        files_info = []
        images_data = []

        for file_path in file_paths:
            file_path_obj = Path(file_path)

            if not file_path_obj.exists():
                files_info.append(f"- {file_path} (FILE NOT FOUND)")
                continue

            file_name = file_path_obj.name
            file_ext = file_path_obj.suffix.lower()
            file_size = file_path_obj.stat().st_size

            # Process different file types
            if file_ext in self.IMAGE_EXTENSIONS:
                file_info, image_data = self._process_image_file(
                    file_path, file_name, file_ext, file_size
                )
                files_info.append(file_info)
                if image_data:
                    images_data.append(image_data)

            elif file_ext in self.EXCEL_EXTENSIONS:
                file_info = self._process_excel_file(
                    file_path, file_name, file_ext, file_size
                )
                files_info.append(file_info)

            elif file_ext in self.TEXT_READABLE_EXTENSIONS:
                file_info = self._process_text_file(
                    file_path, file_name, file_ext, file_size
                )
                files_info.append(file_info)

            else:
                # Other files - just show file name and basic info
                size_str = self._format_file_size(file_size)
                files_info.append(
                    f"- {file_name} ({size_str})\n"
                    f"  Type: {file_ext if file_ext else 'Unknown'} file\n"
                    f"  Path: {file_path}"
                )

        return files_info, images_data

    def generate_workflow_recommendation(self, file_paths, num_prompts=1):
        """Generate workflow recommendations for given file paths using LLM inference.

        Args:
            file_paths: List of file paths or a single file path string
            num_prompts: Number of workflow recommendations to generate (default=1)

        Returns:
            List[str]: List of workflow recommendation strings (always returns a list for consistency)
        """
        # Analyze files
        files_info, images_data = self._analyze_files(file_paths)

        # Load prompt template and format it
        prompt_modifier = self._load_prompt_template(
            "workflow_recommendation_prompt.txt"
        )
        files_info_text = "\n\n".join(files_info)
        formatted_prompt = prompt_modifier.format(files_info=files_info_text)

        # Create message content for LLM
        if images_data:
            message_content = [{"type": "text", "text": formatted_prompt}]
            message_content.extend(images_data)
        else:
            message_content = formatted_prompt

        # Create LLM instance and structured output
        workflow_llm = get_llm(model="us.anthropic.claude-haiku-4-5-20251001-v1:0")
        structured_llm = workflow_llm.with_structured_output(WorkflowRecommendation)

        message = HumanMessage(content=message_content)

        # Generate workflows (single or multiple in parallel)
        if num_prompts == 1:
            workflow_output = structured_llm.invoke([message])
            return [self._format_workflow_output(workflow_output)]
        else:
            messages_list = [[message] for _ in range(num_prompts)]
            workflow_outputs = structured_llm.batch(messages_list)
            return [self._format_workflow_output(output) for output in workflow_outputs]

    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            str: Extracted text from PDF
        """
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

    def _find_section_in_paper(
        self, text, section_name, start_keywords, end_keywords, max_chars=15000
    ):
        """Generic function to find and extract a section from paper text.

        Args:
            text: Full paper text
            section_name: Name of section for logging
            start_keywords: List of keywords that indicate section start
            end_keywords: List of keywords that indicate section end
            max_chars: Maximum characters to return

        Returns:
            str: Extracted section text
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

    def extract_methods_section(self, text):
        """Find and extract the Methods section from paper text.

        Args:
            text: Full paper text

        Returns:
            str: Extracted Methods section
        """
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

        return self._find_section_in_paper(
            text,
            "Methods",
            methods_keywords,
            end_keywords,
        )

    def extract_results_section(self, text):
        """Find and extract the Results section from paper text.

        Args:
            text: Full paper text

        Returns:
            str: Extracted Results section
        """
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

        return self._find_section_in_paper(
            text,
            "Results",
            results_keywords,
            end_keywords,
        )

    def extract_workflow_from_paper(self, pdf_path, mode="integrated"):
        """Extract analysis workflow from PDF paper.

        This method extracts the computational/analytical workflow from a research paper,
        combining information from Results and Methods sections to create a structured
        workflow description.

        Args:
            pdf_path: Path to PDF file
            mode: Extraction mode - "integrated" (Results+Methods), "methods_only", or "results_only"

        Returns:
            str: Structured workflow as numbered list with analysis steps
        """
        # Step 1: Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)

        # Create LLM instance for workflow extraction
        workflow_llm = get_llm(model="us.anthropic.claude-haiku-4-5-20251001-v1:0")

        # Step 2: Extract relevant sections based on mode
        if mode == "integrated":
            results_section = self.extract_results_section(text)
            methods_section = self.extract_methods_section(text)

            prompt_template = self._load_prompt_template(
                "paper_workflow_extraction_integrated.txt"
            )
            prompt = prompt_template.format(
                results_section=results_section, methods_section=methods_section
            )

        elif mode == "methods_only":
            methods_section = self.extract_methods_section(text)

            prompt_template = self._load_prompt_template(
                "paper_workflow_extraction_methods_only.txt"
            )
            prompt = prompt_template.format(methods_section=methods_section)

        elif mode == "results_only":
            results_section = self.extract_results_section(text)

            prompt_template = self._load_prompt_template(
                "paper_workflow_extraction_results_only.txt"
            )
            prompt = prompt_template.format(results_section=results_section)

        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'integrated', 'methods_only', or 'results_only'"
            )

        # Get workflow from LLM
        response = workflow_llm.invoke([HumanMessage(content=prompt)])
        workflow = response.content.strip()

        # Clean up the response
        workflow = re.sub(r"^```.*?\n", "", workflow)
        workflow = re.sub(r"\n```$", "", workflow)

        return workflow

    def generate_workflow_from_paper(
        self, pdf_path, file_paths=None, mode="integrated", num_prompts=1
    ):
        """Generate workflow recommendations based on research paper and optional data files.

        This method combines paper-based workflow extraction with data file analysis
        to generate comprehensive workflow recommendations that are tailored to the
        specific research context and available data.

        Args:
            pdf_path: Path to research paper PDF
            file_paths: Optional list of data file paths to analyze together with paper
            mode: Paper extraction mode - "integrated", "methods_only", or "results_only"
            num_prompts: Number of workflow variations to generate (default=1)

        Returns:
            List[str]: List of workflow recommendation strings
        """
        # Extract workflow from paper
        workflow = self.extract_workflow_from_paper(pdf_path, mode=mode)

        # Build context from paper
        paper_name = os.path.basename(pdf_path)
        context_parts = [
            f"# Research Paper Analysis\n",
            f"**Paper**: {paper_name}\n",
            f"\n## Extracted Workflow\n{workflow}\n",
        ]

        # If data files are provided, analyze them too
        images_data = []
        if file_paths:
            files_info, images_data = self._analyze_files(file_paths)

            if files_info:
                context_parts.append("\n## Available Data Files\n")
                context_parts.append("\n\n".join(files_info))

        # Create comprehensive prompt
        full_context = "".join(context_parts)

        # Load prompt template and format it
        prompt_template = self._load_prompt_template("workflow_from_paper_prompt.txt")
        prompt_text = prompt_template.format(full_context=full_context)

        # Create message content for LLM
        if images_data:
            message_content = [{"type": "text", "text": prompt_text}]
            message_content.extend(images_data)
        else:
            message_content = prompt_text

        # Create LLM instance and structured output
        workflow_llm = get_llm(model="us.anthropic.claude-haiku-4-5-20251001-v1:0")
        structured_llm = workflow_llm.with_structured_output(WorkflowRecommendation)

        message = HumanMessage(content=message_content)

        # Generate workflows (single or multiple in parallel)
        if num_prompts == 1:
            workflow_output = structured_llm.invoke([message])
            return [self._format_workflow_output(workflow_output)]
        else:
            messages_list = [[message] for _ in range(num_prompts)]
            workflow_outputs = structured_llm.batch(messages_list)
            return [self._format_workflow_output(output) for output in workflow_outputs]
