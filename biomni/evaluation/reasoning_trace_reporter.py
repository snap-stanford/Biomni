"""
Reasoning Trace Reporter for Biomni

This module provides functionality to generate detailed HTML reports of biomni's reasoning trace,
including all tool calls, code execution, and reasoning steps for evaluation purposes.
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from jinja2 import Template


class ReasoningTraceReporter:
    """
    A class to generate detailed HTML reports of biomni's reasoning trace.

    This reporter captures:
    - User queries and system responses
    - Tool calls and their parameters
    - Code execution (both generated and called)
    - Reasoning steps and thought processes
    - Execution timing and performance metrics
    """

    def __init__(self, output_dir: str = "evaluation_results/reasoning_traces"):
        """
        Initialize the reasoning trace reporter.

        Args:
            output_dir: Directory to save HTML reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Trace data structure
        self.trace_data = {
            "query": "",
            "start_time": None,
            "end_time": None,
            "steps": [],
            "code_executions": [],
            "performance_metrics": {},
            "complete_terminal_output": [],  # Store complete terminal output
            "generated_plots": [],  # Store generated plots
            "final_result": "",  # Store final result
        }

        # HTML template for the report
        self.html_template = self._get_html_template()

    def start_trace(self, query: str):
        """Start tracing a new query execution."""
        # Create query-specific subfolder
        query_slug = re.sub(r"[^a-zA-Z0-9]", "_", query[:50])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.query_folder = self.output_dir / f"query_{query_slug}_{timestamp}"
        self.query_folder.mkdir(parents=True, exist_ok=True)

        self.trace_data = {
            "query": query,
            "start_time": datetime.now(),
            "end_time": None,
            "steps": [],
            "code_executions": [],
            "performance_metrics": {},
            "complete_terminal_output": [],  # Initialize terminal output storage
            "generated_plots": [],  # Store generated plots
            "final_result": "",  # Store final result
        }

    def add_step(self, step_type: str, content: Any, metadata: dict | None = None):
        """
        Add a step to the reasoning trace.

        Args:
            step_type: Type of step (e.g., 'thinking', 'tool_call', 'code_execution', 'observation')
            content: Content of the step
            metadata: Additional metadata about the step
        """
        step = {
            "type": step_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        self.trace_data["steps"].append(step)

    def add_code_execution(self, code: str, result: Any, execution_time: float = None, is_generated: bool = True):
        """
        Add a code execution to the trace.

        Args:
            code: Code that was executed
            result: Result of the execution
            execution_time: Time taken for execution
            is_generated: Whether the code was generated on-the-fly
        """
        code_execution = {
            "code": code,
            "result": result,
            "execution_time": execution_time,
            "is_generated": is_generated,
            "timestamp": datetime.now().isoformat(),
        }
        self.trace_data["code_executions"].append(code_execution)
        self.trace_data["steps"].append(
            {
                "type": "code_execution",
                "content": code_execution,
                "metadata": {},
                "timestamp": datetime.now().isoformat(),
            }
        )

    def end_trace(self, final_result: Any = None):
        """End the current trace and calculate performance metrics."""
        self.trace_data["end_time"] = datetime.now()

        if self.trace_data["start_time"] and self.trace_data["end_time"]:
            total_time = (self.trace_data["end_time"] - self.trace_data["start_time"]).total_seconds()
            self.trace_data["performance_metrics"]["total_execution_time"] = total_time
            self.trace_data["performance_metrics"]["total_steps"] = len(self.trace_data["steps"])
            self.trace_data["performance_metrics"]["total_code_executions"] = len(self.trace_data["code_executions"])

        if final_result:
            self.trace_data["final_result"] = final_result

    def parse_agent_log(self, log: list[Any]):
        """
        Parse an agent log to extract reasoning trace information.

        Args:
            log: List of log entries from the agent
        """
        for i, log_entry in enumerate(log):
            if isinstance(log_entry, str):
                # Parse different types of log entries
                self._parse_log_entry(log_entry, i)

    def _parse_log_entry(self, log_entry: str, step_index: int):
        """Parse a single log entry to extract trace information."""

        # Clean up the log entry - handle escaped characters and formatting
        cleaned_entry = self._clean_log_entry(log_entry)

        # First, check for structured content (think, execute, solution blocks)
        if "<think>" in cleaned_entry and "</think>" in cleaned_entry:
            self._extract_structured_thinking(cleaned_entry, step_index)
        elif "<execute>" in cleaned_entry and "</execute>" in cleaned_entry:
            self._extract_code_execution(cleaned_entry, step_index)
        elif "<solution>" in cleaned_entry and "</solution>" in cleaned_entry:
            self._extract_solution(cleaned_entry, step_index)

        # Look for run_python_repl tool calls
        elif "run_python_repl" in cleaned_entry:
            self._extract_code_execution(cleaned_entry, step_index)

        # Look for observations (tool results)
        elif "<observation>" in cleaned_entry:
            self._extract_observation(cleaned_entry, step_index)

        # Look for errors (including matplotlib GUI errors)
        elif any(
            error_marker in cleaned_entry
            for error_marker in [
                "NSInternalInconsistencyException",
                "NSWindow",
                "libc++abi: terminating",
                "matplotlib",
                "GUI",
                "thread",
                "abort",
            ]
        ):
            self._extract_error(cleaned_entry, step_index)

        # Look for planning and checklist patterns
        elif any(
            planning_marker in cleaned_entry.lower()
            for planning_marker in ["plan checklist", "updated plan", "step", "checklist", "plan:", "steps:"]
        ):
            self._extract_planning(cleaned_entry, step_index)

        # Look for thinking/reasoning patterns in unstructured text
        elif any(
            thinking_marker in cleaned_entry.lower()
            for thinking_marker in [
                "i need to",
                "let me",
                "first, i'll",
                "i should",
                "i think",
                "i believe",
                "based on",
                "considering",
                "looking at",
                "analyzing",
                "examining",
            ]
        ):
            self._extract_thinking(cleaned_entry, step_index)

        # Look for code blocks that might be generated
        elif "```python" in cleaned_entry or "```" in cleaned_entry:
            self._extract_code_generation(cleaned_entry, step_index)

        # Look for function definitions and code structures
        elif any(
            code_marker in cleaned_entry
            for code_marker in ["def ", "import ", "class ", "for ", "if ", "while ", "try:", "except:"]
        ):
            self._extract_code_generation(cleaned_entry, step_index)

        # Add as general step if not categorized
        else:
            self.add_step("general", cleaned_entry, {"step_index": step_index})

    def _clean_log_entry(self, log_entry: str) -> str:
        """Clean up log entry by handling escaped characters and formatting."""
        import html

        # Decode HTML entities
        cleaned = html.unescape(log_entry)

        # Handle common escaped characters
        cleaned = cleaned.replace("\\n", "\n")
        cleaned = cleaned.replace("\\t", "\t")
        cleaned = cleaned.replace("\\r", "\r")
        cleaned = cleaned.replace("\\'", "'")
        cleaned = cleaned.replace('\\"', '"')

        # Handle unicode escapes
        cleaned = cleaned.replace("\\u0027", "'")
        cleaned = cleaned.replace("\\u0026", "&")
        cleaned = cleaned.replace("\\u003c", "<")
        cleaned = cleaned.replace("\\u003e", ">")

        return cleaned

    def _extract_planning(self, log_entry: str, step_index: int):
        """Extract planning and checklist information from log entry."""
        # Look for planning patterns
        planning_patterns = [
            r"### (?:Updated )?Plan.*?(?=\n\n|\n###|\Z)",
            r"### Plan Checklist.*?(?=\n\n|\n###|\Z)",
            r"### Steps.*?(?=\n\n|\n###|\Z)",
            r"1\. \[.*?\].*?(?=\n\n|\n###|\Z)",
        ]

        for pattern in planning_patterns:
            match = re.search(pattern, log_entry, re.DOTALL | re.IGNORECASE)
            if match:
                planning_content = match.group(0).strip()
                self.add_step(
                    "planning",
                    {"content": planning_content, "type": "planning", "full_context": log_entry},
                    {"step_index": step_index},
                )
                return

        # If no specific pattern found, add as planning if it contains planning keywords
        if any(keyword in log_entry.lower() for keyword in ["plan", "step", "checklist", "next"]):
            self.add_step(
                "planning",
                {"content": log_entry, "type": "planning", "full_context": log_entry},
                {"step_index": step_index},
            )

    def _extract_code_execution(self, log_entry: str, step_index: int):
        """Extract code execution information from log entry."""
        execute_match = re.search(r"<execute>(.*?)</execute>", log_entry, re.DOTALL)
        if execute_match:
            code = execute_match.group(1).strip()

            # Check for matplotlib errors and provide solutions
            execution_result = "Executed via <execute> block"
            if "plt.show()" in code and ("NSWindow" in log_entry or "NSInternalInconsistencyException" in log_entry):
                execution_result = "Executed via <execute> block (matplotlib GUI error - use non-GUI backend)"
            elif "NSWindow" in log_entry or "NSInternalInconsistencyException" in log_entry:
                execution_result = "Executed via <execute> block (GUI threading error)"

            self.add_step(
                "code_execution",
                {"code": code, "is_generated": True, "execution_result": execution_result},
                {"step_index": step_index},
            )

            # Also add to code_executions list for summary
            self.trace_data["code_executions"].append(
                {
                    "code": code,
                    "result": execution_result,
                    "execution_time": None,
                    "is_generated": True,
                    "timestamp": self.trace_data["steps"][-1]["timestamp"],
                }
            )
        elif "run_python_repl" in log_entry:
            # Extract code from run_python_repl tool calls
            code_match = re.search(r'command["\']?\s*:\s*["\']([^"\']+)["\']', log_entry)
            if code_match:
                code = code_match.group(1).strip()

                # Check for matplotlib errors
                execution_result = "Executed via run_python_repl"
                if "plt.show()" in code and (
                    "NSWindow" in log_entry or "NSInternalInconsistencyException" in log_entry
                ):
                    execution_result = "Executed via run_python_repl (matplotlib GUI error - use non-GUI backend)"

                self.add_step(
                    "code_execution",
                    {
                        "code": code,
                        "is_generated": True,
                        "tool": "run_python_repl",
                        "execution_result": execution_result,
                    },
                    {"step_index": step_index},
                )

                # Also add to code_executions list for summary
                self.trace_data["code_executions"].append(
                    {
                        "code": code,
                        "result": execution_result,
                        "execution_time": None,
                        "is_generated": True,
                        "timestamp": self.trace_data["steps"][-1]["timestamp"],
                    }
                )

    def _extract_observation(self, log_entry: str, step_index: int):
        """Extract observation (tool result) information from log entry."""
        obs_match = re.search(r"<observation>(.*?)</observation>", log_entry, re.DOTALL)
        if obs_match:
            result = obs_match.group(1).strip()
            self.add_step(
                "observation",
                {"content": result, "type": "tool_result", "full_context": log_entry},
                {"step_index": step_index},
            )
        else:
            # Look for other observation patterns
            if "result:" in log_entry.lower() or "output:" in log_entry.lower():
                self.add_step(
                    "observation",
                    {"content": log_entry, "type": "general_result", "full_context": log_entry},
                    {"step_index": step_index},
                )
            else:
                self.add_step(
                    "observation",
                    {"content": log_entry, "type": "general", "full_context": log_entry},
                    {"step_index": step_index},
                )

    def _extract_structured_thinking(self, log_entry: str, step_index: int):
        """Extract structured thinking from <think> tags."""
        think_match = re.search(r"<think>(.*?)</think>", log_entry, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            self.add_step(
                "thinking",
                {"content": thinking, "type": "structured", "full_context": log_entry},
                {"step_index": step_index},
            )
        else:
            self.add_step(
                "thinking",
                {"content": log_entry, "type": "unstructured", "full_context": log_entry},
                {"step_index": step_index},
            )

    def _extract_thinking(self, log_entry: str, step_index: int):
        """Extract thinking/reasoning information from unstructured log entry."""
        self.add_step(
            "thinking",
            {"content": log_entry, "type": "unstructured", "full_context": log_entry},
            {"step_index": step_index},
        )

    def _extract_solution(self, log_entry: str, step_index: int):
        """Extract solution information from <solution> tags."""
        solution_match = re.search(r"<solution>(.*?)</solution>", log_entry, re.DOTALL)
        if solution_match:
            solution = solution_match.group(1).strip()
            self.add_step("solution", {"content": solution, "full_context": log_entry}, {"step_index": step_index})
        else:
            self.add_step("solution", {"content": log_entry, "full_context": log_entry}, {"step_index": step_index})

    def _extract_error(self, log_entry: str, step_index: int):
        """Extract error information from log entry."""
        error_type = "Unknown Error"
        error_message = log_entry

        if "NSInternalInconsistencyException" in log_entry and "NSWindow" in log_entry:
            error_type = "Matplotlib GUI Error"
            error_message = "GUI window creation failed - use non-GUI backend (e.g., 'Agg')"
        elif "matplotlib" in log_entry.lower():
            error_type = "Matplotlib Error"
        elif "thread" in log_entry.lower():
            error_type = "Threading Error"
        elif "abort" in log_entry.lower():
            error_type = "Process Abort"

        self.add_step(
            "error",
            {"error_type": error_type, "error_message": error_message, "full_log": log_entry},
            {"step_index": step_index},
        )

    def _extract_code_generation(self, log_entry: str, step_index: int):
        """Extract code generation information from log entry."""
        # Look for Python code blocks
        code_match = re.search(r"```python\s*(.*?)\s*```", log_entry, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            self.add_step(
                "code_generation",
                {"code": code, "is_generated": True, "context": log_entry},
                {"step_index": step_index},
            )
        else:
            # Look for any code block
            code_match = re.search(r"```\s*(.*?)\s*```", log_entry, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                self.add_step(
                    "code_generation",
                    {"code": code, "is_generated": True, "context": log_entry},
                    {"step_index": step_index},
                )
            else:
                self.add_step(
                    "code_generation",
                    {"code": log_entry, "is_generated": True, "context": "Code generation step"},
                    {"step_index": step_index},
                )

    def generate_html_report(self, filename: str | None = None) -> str:
        """
        Generate an HTML report from the current trace data.

        Args:
            filename: Optional filename for the report

        Returns:
            Path to the generated HTML file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            query_slug = re.sub(r"[^a-zA-Z0-9]", "_", self.trace_data["query"][:50])
            filename = f"reasoning_trace_{query_slug}_{timestamp}.html"

        filepath = self.query_folder / filename

        # Prepare data for template
        template_data = {
            "query": self.trace_data["query"],
            "start_time": self.trace_data["start_time"].isoformat() if self.trace_data["start_time"] else "",
            "end_time": self.trace_data["end_time"].isoformat() if self.trace_data["end_time"] else "",
            "total_steps": len(self.trace_data["steps"]),
            "total_code_executions": len(self.trace_data["code_executions"]),
            "performance_metrics": self.trace_data["performance_metrics"],
            "steps": self.trace_data["steps"],
            "code_executions": self.trace_data["code_executions"],
            "final_result": self.trace_data.get("final_result", ""),
            "generated_plots": self.trace_data.get("generated_plots", []),
        }

        # Render template
        template = Template(self.html_template)
        html_content = template.render(**template_data)

        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        return str(filepath)

    def add_terminal_output(self, output: str, output_type: str = "general"):
        """
        Add terminal output to the trace.

        Args:
            output: The terminal output text
            output_type: Type of output (e.g., 'planning', 'execution', 'result', 'error')
        """
        terminal_entry = {"content": output, "type": output_type, "timestamp": datetime.now().isoformat()}
        self.trace_data["complete_terminal_output"].append(terminal_entry)

    def save_complete_terminal_output(self, filename: str | None = None) -> str:
        """
        Save the complete terminal output to a text file.

        Args:
            filename: Optional filename for the output file

        Returns:
            Path to the saved text file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            query_slug = re.sub(r"[^a-zA-Z0-9]", "_", self.trace_data["query"][:50])
            filename = f"complete_output_{query_slug}_{timestamp}.txt"

        filepath = self.query_folder / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("COMPLETE TERMINAL OUTPUT - BIOMNI REASONING TRACE\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Query: {self.trace_data['query']}\n")
            f.write(f"Start Time: {self.trace_data['start_time']}\n")
            f.write(f"End Time: {self.trace_data['end_time']}\n")
            f.write(
                f"Total Execution Time: {self.trace_data['performance_metrics'].get('total_execution_time', 'N/A')} seconds\n"
            )
            f.write("\n" + "=" * 80 + "\n\n")

            for i, output_entry in enumerate(self.trace_data["complete_terminal_output"], 1):
                f.write(f"[{i}] {output_entry['timestamp']} - {output_entry['type'].upper()}\n")
                f.write("-" * 40 + "\n")
                f.write(output_entry["content"])
                f.write("\n\n")

        return str(filepath)

    def capture_plot(self, plot_name: str = None):
        """
        Capture the current matplotlib plot and save it.

        Args:
            plot_name: Optional name for the plot file
        """
        try:
            import matplotlib.pyplot as plt

            if not plot_name:
                plot_name = f"plot_{len(self.trace_data['generated_plots']) + 1}"

            # Save plot to query folder
            plot_path = self.query_folder / f"{plot_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")

            # Store plot information
            plot_info = {"name": plot_name, "path": str(plot_path), "timestamp": datetime.now().isoformat()}
            self.trace_data["generated_plots"].append(plot_info)

            return str(plot_path)

        except Exception as e:
            print(f"Warning: Could not capture plot: {e}")
            return None

    def set_final_result(self, result: str):
        """Set the final result for the query."""
        self.trace_data["final_result"] = result

    def generate_final_user_report(self, filename: str | None = None) -> str:
        """
        Generate a clean, final user report with plots and evidence.

        Args:
            filename: Optional filename for the report

        Returns:
            Path to the generated HTML file
        """
        if not filename:
            filename = "final_user_report.html"

        filepath = self.query_folder / filename

        # Generate the HTML content
        html_content = self._get_final_report_template()

        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        return str(filepath)

    def _get_final_report_template(self) -> str:
        """Get the HTML template for the final user report."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Final Report - {self.trace_data["query"][:50]}...</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            }}
        }};
    </script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            margin-bottom: 40px;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #667eea;
        }}
        .section h2 {{
            color: #333;
            margin-top: 0;
            font-size: 1.8em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .query-box {{
            background: #e3f2fd;
            border: 2px solid #2196f3;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            font-size: 1.1em;
            line-height: 1.6;
        }}
        .result-box {{
            background: #e8f5e8;
            border: 2px solid #4caf50;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            font-size: 1.1em;
            line-height: 1.6;
        }}
        .result-box h1, .result-box h2, .result-box h3, .result-box h4, .result-box h5, .result-box h6 {{
            color: #2e7d32;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .result-box p {{
            margin-bottom: 15px;
        }}
        .result-box ul, .result-box ol {{
            margin-bottom: 15px;
            padding-left: 20px;
        }}
        .result-box li {{
            margin-bottom: 5px;
        }}
        .result-box code {{
            background: #f0f0f0;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        .result-box pre {{
            background: #f4f4f4;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
            margin: 15px 0;
        }}
        .result-box blockquote {{
            border-left: 4px solid #4caf50;
            margin: 15px 0;
            padding-left: 15px;
            color: #666;
        }}
        .plot-container {{
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        .plot-title {{
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-top: 4px solid #667eea;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .footer {{
            background: #333;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
        .code-block {{
            background: #f4f4f4;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 Final Analysis Report</h1>
            <p>Comprehensive analysis and findings from biomni reasoning system</p>
        </div>

        <div class="content">
            <div class="section">
                <h2>📋 Query</h2>
                <div class="query-box">
                    {self.trace_data["query"]}
                </div>
            </div>

            <div class="section">
                <h2>📊 Analysis Summary</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{len(self.trace_data["steps"])}</div>
                        <div class="metric-label">Total Steps</div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-value">{len(self.trace_data["code_executions"])}</div>
                        <div class="metric-label">Code Executions</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(self.trace_data["generated_plots"])}</div>
                        <div class="metric-label">Generated Plots</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>🎯 Final Results</h2>
                <div class="result-box">
                    {self._render_markdown_content(self.trace_data["final_result"]) if self.trace_data["final_result"] else "Results will be displayed here after analysis completion."}
                </div>
            </div>

            <div class="section">
                <h2>📈 Generated Visualizations</h2>
                {self._generate_plots_html()}
            </div>
        </div>

        <div class="footer">
            <p>Generated by biomni reasoning system on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Execution time: {self.trace_data["performance_metrics"].get("total_execution_time", "N/A")} seconds</p>
        </div>
    </div>
</body>
</html>
        """

    def _generate_plots_html(self) -> str:
        """Generate HTML for plots section."""
        if not self.trace_data["generated_plots"]:
            return "<p>No plots were generated during this analysis.</p>"

        plots_html = ""
        for plot in self.trace_data["generated_plots"]:
            # Use relative path for the plot image
            plot_filename = os.path.basename(plot["path"])
            plots_html += f"""
            <div class="plot-container">
                <div class="plot-title">{plot["name"].replace("_", " ").title()}</div>
                <img src="{plot_filename}" alt="{plot["name"]}" />
                <p><em>Generated at: {plot["timestamp"]}</em></p>
            </div>
            """

        return plots_html

    def _render_markdown_content(self, content: str) -> str:
        """
        Convert markdown content to HTML with proper formatting.

        Args:
            content: Markdown content to render

        Returns:
            HTML content with proper formatting
        """
        if not content:
            return ""

        # Basic markdown to HTML conversion
        html = content

        # Headers
        html = re.sub(r"^### (.*?)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.*?)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^# (.*?)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)

        # Bold and italic
        html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"\*(.*?)\*", r"<em>\1</em>", html)

        # Code blocks
        html = re.sub(r"```(.*?)```", r"<pre><code>\1</code></pre>", html, flags=re.DOTALL)
        html = re.sub(r"`(.*?)`", r"<code>\1</code>", html)

        # Lists
        html = re.sub(r"^\d+\. (.*?)$", r"<li>\1</li>", html, flags=re.MULTILINE)
        html = re.sub(r"^- (.*?)$", r"<li>\1</li>", html, flags=re.MULTILINE)

        # Wrap consecutive list items in <ol> or <ul>
        html = re.sub(r"(<li>.*?</li>\n?)+", lambda m: f"<ol>{m.group(0)}</ol>", html, flags=re.DOTALL)

        # Line breaks
        html = re.sub(r"\n\n", r"</p><p>", html)
        html = re.sub(r"\n", r"<br>", html)

        # Wrap in paragraphs if not already wrapped
        if not html.startswith("<"):
            html = f"<p>{html}</p>"

        # Clean up empty paragraphs
        html = re.sub(r"<p>\s*</p>", "", html)
        html = re.sub(r"<p><br></p>", "", html)

        return html

    def _get_html_template(self) -> str:
        """Get the HTML template for the report."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Biomni Reasoning Trace Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header .subtitle {
            margin-top: 10px;
            opacity: 0.9;
            font-size: 1.1em;
        }
        .content {
            padding: 30px;
        }
        .section {
            margin-bottom: 40px;
        }
        .section h2 {
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            color: #666;
            margin-top: 5px;
        }
        .query-box {
            background: #e3f2fd;
            border: 1px solid #2196f3;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
        }
        .query-box h3 {
            margin-top: 0;
            color: #1976d2;
        }
        .step {
            margin-bottom: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }
        .step-header {
            background: #f5f5f5;
            padding: 15px;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .step-type {
            background: #667eea;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .step-timestamp {
            color: #666;
            font-size: 0.9em;
        }
        .step-content {
            padding: 20px;
        }
        .code-block {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            overflow-x: auto;
        }
        .tool-call {
            background: #fff3e0;
            border: 1px solid #ff9800;
            border-radius: 4px;
            padding: 15px;
            margin: 10px 0;
        }
        .thinking {
            background: #e8f5e8;
            border: 1px solid #4caf50;
            border-radius: 4px;
            padding: 15px;
            margin: 10px 0;
        }
                 .observation {
             background: #e3f2fd;
             border: 1px solid #2196f3;
             border-radius: 4px;
             padding: 15px;
             margin: 10px 0;
         }
         .error {
             background: #ffebee;
             border: 1px solid #f44336;
             border-radius: 4px;
             padding: 15px;
             margin: 10px 0;
         }
         .error h4 {
             color: #d32f2f;
             margin-top: 0;
         }
         .solution {
             background: #e8f5e8;
             border: 1px solid #4caf50;
             border-radius: 4px;
             padding: 15px;
             margin: 10px 0;
         }
         .solution h4 {
             color: #2e7d32;
             margin-top: 0;
         }
         .planning {
             background: #fff3e0;
             border: 1px solid #ff9800;
             border-radius: 4px;
             padding: 15px;
             margin: 10px 0;
         }
         .planning h4 {
             color: #e65100;
             margin-top: 0;
         }
         .planning-content {
             white-space: pre-line;
             font-family: 'Courier New', monospace;
             background: #f5f5f5;
             padding: 10px;
             border-radius: 3px;
             margin: 10px 0;
         }
        .final-result {
            background: #f3e5f5;
            border: 1px solid #9c27b0;
            border-radius: 8px;
            padding: 20px;
            margin-top: 30px;
        }
        .final-result h3 {
            margin-top: 0;
            color: #7b1fa2;
        }
        .collapsible {
            cursor: pointer;
        }
        .collapsible:hover {
            background-color: #f0f0f0;
        }
        .collapsible-content {
            display: none;
            padding: 15px;
            background-color: #f9f9f9;
        }
        .show {
            display: block;
        }
        .json-viewer {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            overflow-x: auto;
            max-height: 300px;
            overflow-y: auto;
        }


    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 Biomni Reasoning Trace Report</h1>
            <div class="subtitle">Detailed analysis of AI reasoning and tool execution</div>
        </div>

        <div class="content">
            <!-- Query Section -->
            <div class="section">
                <div class="query-box">
                    <h3>🔍 User Query</h3>
                    <p>{{ query }}</p>
                </div>
            </div>

            <!-- Performance Metrics -->
            <div class="section">
                <h2>📊 Performance Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{{ total_steps }}</div>
                        <div class="metric-label">Total Steps</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ total_code_executions }}</div>
                        <div class="metric-label">Code Executions</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(performance_metrics.total_execution_time) if performance_metrics.total_execution_time else "N/A" }}</div>
                        <div class="metric-label">Execution Time (s)</div>
                    </div>
                </div>

                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{{ start_time[:19] if start_time else "N/A" }}</div>
                        <div class="metric-label">Start Time</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ end_time[:19] if end_time else "N/A" }}</div>
                        <div class="metric-label">End Time</div>
                    </div>
                </div>
            </div>

            <!-- Detailed Steps -->
            <div class="section">
                <h2>🔄 Reasoning Steps</h2>
                {% for step in steps %}
                <div class="step">
                    <div class="step-header collapsible" onclick="toggleStep({{ loop.index }})">
                        <div>
                            <span class="step-type">{{ step.type.upper() }}</span>
                            <span style="margin-left: 10px;">Step {{ loop.index }}</span>
                        </div>
                        <div class="step-timestamp">{{ step.timestamp[:19] }}</div>
                    </div>
                    <div class="step-content collapsible-content" id="step-{{ loop.index }}">
                        {% if step.type == 'code_execution' %}
                             <div>
                                 <h4>💻 Code Execution {% if step.content.is_generated %}(Generated){% endif %}</h4>
                                 <div class="code-block">{{ step.content.code }}</div>
                                 {% if step.content.tool %}
                                 <p><strong>Tool:</strong> {{ step.content.tool }}</p>
                                 {% endif %}
                             </div>
                         {% elif step.type == 'code_generation' %}
                             <div>
                                 <h4>📝 Code Generation</h4>
                                 <div class="code-block">{{ step.content.code }}</div>
                                 {% if step.content.context %}
                                 <p><strong>Context:</strong> {{ step.content.context[:200] }}{% if step.content.context|length > 200 %}...{% endif %}</p>
                                 {% endif %}
                             </div>
                                                  {% elif step.type == 'thinking' %}
                             <div class="thinking">
                                 <h4>🧠 Reasoning {% if step.content.type == 'structured' %}(Structured){% else %}(Unstructured){% endif %}</h4>
                                 {% if step.content.content %}
                                     <p>{{ step.content.content }}</p>
                                 {% else %}
                                     <p>{{ step.content }}</p>
                                 {% endif %}
                                 {% if step.content.full_context and step.content.full_context != step.content.content %}
                                 <details>
                                     <summary>Full Context</summary>
                                     <div class="code-block">{{ step.content.full_context }}</div>
                                 </details>
                                 {% endif %}
                             </div>
                         {% elif step.type == 'solution' %}
                             <div class="solution">
                                 <h4>✅ Solution</h4>
                                 <p>{{ step.content.content }}</p>
                                 {% if step.content.full_context and step.content.full_context != step.content.content %}
                                 <details>
                                     <summary>Full Context</summary>
                                     <div class="code-block">{{ step.content.full_context }}</div>
                                 </details>
                                 {% endif %}
                             </div>
                         {% elif step.type == 'planning' %}
                             <div class="planning">
                                 <h4>📋 Planning</h4>
                                 {% if step.content.content %}
                                     <div class="planning-content">{{ step.content.content | safe }}</div>
                                 {% else %}
                                     <p>{{ step.content }}</p>
                                 {% endif %}
                                 {% if step.content.full_context and step.content.full_context != step.content.content %}
                                 <details>
                                     <summary>Full Context</summary>
                                     <div class="code-block">{{ step.content.full_context }}</div>
                                 </details>
                                 {% endif %}
                             </div>
                         {% elif step.type == 'observation' %}
                             <div class="observation">
                                 <h4>👁️ Observation {% if step.content.type %}({{ step.content.type }}){% endif %}</h4>
                                 {% if step.content.content %}
                                     <p>{{ step.content.content }}</p>
                                 {% else %}
                                     <p>{{ step.content }}</p>
                                 {% endif %}
                                 {% if step.content.full_context and step.content.full_context != step.content.content %}
                                 <details>
                                     <summary>Full Context</summary>
                                     <div class="code-block">{{ step.content.full_context }}</div>
                                 </details>
                                 {% endif %}
                             </div>
                         {% elif step.type == 'error' %}
                             <div class="error">
                                 <h4>❌ {{ step.content.error_type }}</h4>
                                 <p><strong>Error:</strong> {{ step.content.error_message }}</p>
                                 {% if step.content.full_log %}
                                 <details>
                                     <summary>Full Error Log</summary>
                                     <div class="code-block">{{ step.content.full_log }}</div>
                                 </details>
                                 {% endif %}
                             </div>
                        {% else %}
                            <div>
                                <h4>📝 {{ step.type.title() }}</h4>
                                <p>{{ step.content }}</p>
                            </div>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
            </div>





        </div>
    </div>

    <script>
        function toggleStep(stepIndex) {
            const content = document.getElementById('step-' + stepIndex);
            content.classList.toggle('show');
        }

        // Auto-expand first few steps
        window.onload = function() {
            for (let i = 1; i <= Math.min(3, {{ total_steps }}); i++) {
                const content = document.getElementById('step-' + i);
                if (content) {
                    content.classList.add('show');
                }
            }
        };
    </script>
</body>
</html>
        """
