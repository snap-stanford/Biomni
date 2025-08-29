"""
Enhanced A1 Agent with Reasoning Trace Reporting

This module extends the A1 agent with comprehensive reasoning trace reporting capabilities
for detailed evaluation of biomni's reasoning process.
"""

import io
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from biomni.agent.a1 import A1
from biomni.evaluation.reasoning_trace_reporter import ReasoningTraceReporter


class A1WithTrace(A1):
    """
    Enhanced A1 agent with reasoning trace reporting capabilities.
    
    This agent extends the base A1 agent to provide detailed HTML reports
    of the reasoning process, including all tool calls, code execution,
    and reasoning steps.
    """
    
    def __init__(
        self,
        path: str | None = None,
        llm: str | None = None,
        source: str | None = None,
        use_tool_retriever: bool | None = None,
        timeout_seconds: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        enable_trace_reporting: bool = True,
        trace_output_dir: str = "evaluation_results/reasoning_traces",
    ):
        """
        Initialize the enhanced A1 agent with trace reporting.
        
        Args:
            path: Path to the data
            llm: LLM to use for the agent
            source: Source provider for the LLM
            use_tool_retriever: If True, use a tool retriever
            timeout_seconds: Timeout for code execution in seconds
            base_url: Base URL for custom model serving
            api_key: API key for the custom LLM
            enable_trace_reporting: Whether to enable trace reporting
            trace_output_dir: Directory to save trace reports
        """
        # Initialize the base A1 agent
        super().__init__(
            path=path,
            llm=llm,
            source=source,
            use_tool_retriever=use_tool_retriever,
            timeout_seconds=timeout_seconds,
            base_url=base_url,
            api_key=api_key,
        )
        
        # Initialize trace reporting
        self.enable_trace_reporting = enable_trace_reporting
        if self.enable_trace_reporting:
            self.trace_reporter = ReasoningTraceReporter(trace_output_dir)
        else:
            self.trace_reporter = None
        
        # Enhanced logging for trace analysis
        self.enhanced_log = []
        self.current_query = None
        self.execution_start_time = None
        
        # Terminal output capture
        self.terminal_output_buffer = []
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
    
    def go(self, prompt: str) -> Tuple[List[Any], str]:
        """
        Execute the agent with the given prompt and generate trace report.
        
        Args:
            prompt: The user's query
            
        Returns:
            Tuple of (log, final_message_content)
        """
        # Start trace reporting
        if self.enable_trace_reporting:
            self.trace_reporter.start_trace(prompt)
            self.current_query = prompt
            self.execution_start_time = time.time()
            self.enhanced_log = []
            
            # Add initial query to terminal output
            self.trace_reporter.add_terminal_output(f"Query: {prompt}", "query")
        
        # Start terminal capture
        self._start_terminal_capture()
        
        try:
            # Execute the base agent
            log, final_content = super().go(prompt)
            
            # Process trace reporting
            if self.enable_trace_reporting:
                self._process_trace_reporting(log, final_content)
            
            return log, final_content
            
        finally:
            # Stop terminal capture
            self._stop_terminal_capture()
    
    def _start_terminal_capture(self):
        """Start capturing terminal output."""
        if self.enable_trace_reporting:
            self.terminal_output_buffer = []
            # Create a custom stdout that captures output
            class CapturingStdout:
                def __init__(self, original_stdout, buffer, reporter):
                    self.original_stdout = original_stdout
                    self.buffer = buffer
                    self.reporter = reporter
                
                def write(self, text):
                    self.original_stdout.write(text)
                    self.buffer.append(text)
                    if self.reporter and hasattr(self.reporter, 'add_terminal_output'):
                        self.reporter.add_terminal_output(text, "stdout")
                
                def flush(self):
                    self.original_stdout.flush()
            
            sys.stdout = CapturingStdout(self.original_stdout, self.terminal_output_buffer, self.trace_reporter)
    
    def _stop_terminal_capture(self):
        """Stop capturing terminal output."""
        if self.enable_trace_reporting:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
    
    def go_stream(self, prompt: str):
        """
        Execute the agent with streaming and trace reporting.
        
        Args:
            prompt: The user's query
            
        Yields:
            Generator yielding each step of execution
        """
        # Start trace reporting
        if self.enable_trace_reporting:
            self.trace_reporter.start_trace(prompt)
            self.current_query = prompt
            self.execution_start_time = time.time()
            self.enhanced_log = []
        
        # Execute the base agent with streaming
        for step in super().go_stream(prompt):
            # Add to enhanced log for trace analysis
            if self.enable_trace_reporting:
                self.enhanced_log.append(step["output"])
            
            yield step
        
        # Process trace reporting after completion
        if self.enable_trace_reporting:
            self._process_trace_reporting(self.enhanced_log, step.get("output", ""))
    
    def _process_trace_reporting(self, log: List[Any], final_content: str):
        """
        Process the execution log to generate trace report.
        
        Args:
            log: The execution log from the agent
            final_content: The final content returned by the agent
        """
        if not self.enable_trace_reporting or not self.trace_reporter:
            return
        
        # Parse the log to extract trace information
        self.trace_reporter.parse_agent_log(log)
        
        # Add performance metrics
        if self.execution_start_time:
            execution_time = time.time() - self.execution_start_time
            self.trace_reporter.trace_data["performance_metrics"]["total_execution_time"] = execution_time
        

        
        # End the trace
        self.trace_reporter.end_trace(final_content)
    
    def generate_trace_report(self, filename: Optional[str] = None) -> str:
        """
        Generate an HTML trace report for the last execution.
        
        Args:
            filename: Optional filename for the report
            
        Returns:
            Path to the generated HTML file
        """
        if not self.enable_trace_reporting or not self.trace_reporter:
            raise RuntimeError("Trace reporting is not enabled")
        
        return self.trace_reporter.generate_html_report(filename)
    
    def generate_final_user_report(self, filename: Optional[str] = None) -> str:
        """
        Generate a clean, final user report with plots and evidence.
        
        Args:
            filename: Optional filename for the report
            
        Returns:
            Path to the generated HTML file
        """
        if not self.enable_trace_reporting or not self.trace_reporter:
            raise RuntimeError("Trace reporting is not enabled")
        
        return self.trace_reporter.generate_final_user_report(filename)
    
    def capture_plot(self, plot_name: str = None) -> str:
        """
        Capture the current matplotlib plot and save it.
        
        Args:
            plot_name: Optional name for the plot file
            
        Returns:
            Path to the saved plot file
        """
        if not self.enable_trace_reporting or not self.trace_reporter:
            raise RuntimeError("Trace reporting is not enabled")
        
        return self.trace_reporter.capture_plot(plot_name)
    
    def set_final_result(self, result: str):
        """
        Set the final result for the query.
        
        Args:
            result: The final result text
        """
        if not self.enable_trace_reporting or not self.trace_reporter:
            raise RuntimeError("Trace reporting is not enabled")
        
        self.trace_reporter.set_final_result(result)
    

    
    def capture_current_plot(self, plot_name: str = None) -> str:
        """
        Capture the current plot and save it to the query folder.
        This method can be called from within the agent's execution.
        
        Args:
            plot_name: Optional name for the plot file
            
        Returns:
            Path to the saved plot file
        """
        if not self.enable_trace_reporting or not self.trace_reporter:
            return None
        
        try:
            import matplotlib.pyplot as plt
            
            # Check if there's a current figure
            if not plt.get_fignums():
                print("Warning: No active plot to capture")
                return None
            
            # Capture the current figure
            if not plot_name:
                plot_name = f"plot_{len(self.trace_reporter.trace_data['generated_plots']) + 1}"
            
            return self.capture_plot(plot_name)
            
        except Exception as e:
            print(f"Warning: Could not capture current plot: {e}")
            return None
    
    def save_complete_terminal_output(self, filename: Optional[str] = None) -> str:
        """
        Save the complete terminal output to a text file.
        
        Args:
            filename: Optional filename for the output file
            
        Returns:
            Path to the saved text file
        """
        if not self.enable_trace_reporting or not self.trace_reporter:
            raise RuntimeError("Trace reporting is not enabled")
        
        return self.trace_reporter.save_complete_terminal_output(filename)
    
    def get_trace_data(self) -> Dict[str, Any]:
        """
        Get the current trace data for analysis.
        
        Returns:
            Dictionary containing trace data
        """
        if not self.enable_trace_reporting or not self.trace_reporter:
            return {}
        
        return self.trace_reporter.trace_data
    
    def add_custom_trace_step(self, step_type: str, content: Any, metadata: Optional[Dict] = None):
        """
        Add a custom step to the trace for additional analysis.
        
        Args:
            step_type: Type of the step
            content: Content of the step
            metadata: Additional metadata
        """
        if self.enable_trace_reporting and self.trace_reporter:
            self.trace_reporter.add_step(step_type, content, metadata)
    
    def analyze_tool_usage_patterns(self) -> Dict[str, Any]:
        """
        Analyze tool usage patterns from the trace data.
        
        Returns:
            Dictionary containing analysis results
        """
        if not self.enable_trace_reporting or not self.trace_reporter:
            return {}
        
        trace_data = self.trace_reporter.trace_data
        analysis = {
            "total_tool_calls": len(trace_data["tool_calls"]),
            "tool_usage_frequency": {},
            "code_execution_frequency": {},
            "reasoning_steps_breakdown": {},
            "performance_metrics": trace_data["performance_metrics"]
        }
        
        # Analyze tool usage frequency
        for tool_call in trace_data["tool_calls"]:
            tool_name = tool_call["tool_name"]
            analysis["tool_usage_frequency"][tool_name] = analysis["tool_usage_frequency"].get(tool_name, 0) + 1
        
        # Analyze code execution patterns
        for code_exec in trace_data["code_executions"]:
            code_type = "generated" if code_exec["is_generated"] else "pre_written"
            analysis["code_execution_frequency"][code_type] = analysis["code_execution_frequency"].get(code_type, 0) + 1
        
        # Analyze reasoning steps
        for step in trace_data["steps"]:
            step_type = step["type"]
            analysis["reasoning_steps_breakdown"][step_type] = analysis["reasoning_steps_breakdown"].get(step_type, 0) + 1
            
            # Count code generation steps separately
            if step_type == "code_generation":
                analysis["code_execution_frequency"]["generated"] = analysis["code_execution_frequency"].get("generated", 0) + 1
        
        return analysis
    
    def export_trace_data(self, format: str = "json", filename: Optional[str] = None) -> str:
        """
        Export trace data in various formats for further analysis.
        
        Args:
            format: Export format ('json', 'csv', 'pickle')
            filename: Optional filename for export
            
        Returns:
            Path to the exported file
        """
        if not self.enable_trace_reporting or not self.trace_reporter:
            raise RuntimeError("Trace reporting is not enabled")
        
        import json
        import pandas as pd
        import pickle
        from datetime import datetime
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trace_data_{timestamp}.{format}"
        
        # Use query_folder if available, otherwise fall back to output_dir
        if hasattr(self.trace_reporter, 'query_folder') and self.trace_reporter.query_folder:
            filepath = self.trace_reporter.query_folder / filename
        else:
            filepath = self.trace_reporter.output_dir / filename
        
        if format == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.trace_reporter.trace_data, f, indent=2, default=str)
        elif format == "csv":
            # Convert steps to DataFrame
            steps_df = pd.DataFrame(self.trace_reporter.trace_data["steps"])
            steps_df.to_csv(filepath, index=False)
        elif format == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(self.trace_reporter.trace_data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(filepath)

    def _inject_custom_functions_to_repl(self):
        """Override to inject custom functions including plot saving to query folder."""
        # Call the parent method first
        super()._inject_custom_functions_to_repl()
        
        # Add custom plot saving function if trace reporting is enabled
        if self.enable_trace_reporting and self.trace_reporter:
            from biomni.tool.support_tools import _persistent_namespace
            import builtins
            import matplotlib.pyplot as plt
            from datetime import datetime
            
            # Store original savefig if not already stored
            if not hasattr(plt, '_original_savefig'):
                plt._original_savefig = plt.savefig
            
            # Override plt.savefig to save to query folder
            def custom_savefig(*args, **kwargs):
                """
                Override plt.savefig to save to query folder when trace reporting is enabled.
                This ensures all plots are saved in the query-specific directory.
                """
                if self.enable_trace_reporting and self.trace_reporter:
                    # Determine filename
                    if args and isinstance(args[0], str):
                        filename = args[0]
                        other_args = args[1:]
                    else:
                        # Generate filename based on number of existing plots
                        plot_count = len(self.trace_reporter.trace_data['generated_plots']) + 1
                        filename = f"plot_{plot_count}.png"
                        other_args = args
                    
                    # Ensure filename has .png extension
                    if not filename.endswith('.png'):
                        filename += '.png'
                    
                    # Save to query folder
                    plot_path = self.trace_reporter.query_folder / filename
                    result = plt._original_savefig(plot_path, *other_args, **kwargs)
                    
                    # Add to generated plots list for final report
                    plot_info = {
                        "name": filename.replace('.png', ''),
                        "path": str(plot_path),
                        "timestamp": datetime.now().isoformat()
                    }
                    self.trace_reporter.trace_data["generated_plots"].append(plot_info)
                    
                    print(f"Plot saved to query folder: {plot_path}")
                    return result
                else:
                    # Fall back to original savefig
                    return plt._original_savefig(*args, **kwargs)
            
            # Replace plt.savefig with our custom version
            plt.savefig = custom_savefig
            
            # Inject the modified plt into the namespace
            _persistent_namespace['plt'] = plt
            
            # Also provide a convenience function
            def save_plot_to_query_folder(filename=None, **kwargs):
                """
                Convenience function to save plot to query folder.
                This is equivalent to plt.savefig() when trace reporting is enabled.
                
                Args:
                    filename: Optional filename for the plot
                    **kwargs: Additional arguments to pass to plt.savefig()
                """
                if filename:
                    return plt.savefig(filename, **kwargs)
                else:
                    return plt.savefig(**kwargs)
            
            # Inject the convenience function
            _persistent_namespace['save_plot_to_query_folder'] = save_plot_to_query_folder
            
            # Also make it available in builtins
            if not hasattr(builtins, "_biomni_custom_functions"):
                builtins._biomni_custom_functions = {}
            builtins._biomni_custom_functions['save_plot_to_query_folder'] = save_plot_to_query_folder
