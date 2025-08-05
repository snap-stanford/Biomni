import sys
import traceback
from io import StringIO

# Create a persistent namespace that will be shared across all executions
_persistent_namespace = {}


def run_python_repl(command: str) -> str:
    """Executes the provided Python command in a persistent environment and returns the output.
    Variables defined in one execution will be available in subsequent executions.
    """

    def execute_in_repl(command: str) -> str:
        """Helper function to execute the command in the persistent environment."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        # Use the persistent namespace
        global _persistent_namespace

        try:
            # Execute the command in the persistent namespace
            exec(command, _persistent_namespace)
            output = mystdout.getvalue()
        except Exception as e:
            # Get detailed error information with proper line tracking
            exc_type, exc_value, exc_traceback = sys.exc_info()

            # Split command into lines for better error reporting
            command_lines = command.split("\n")

            # Extract detailed traceback information
            tb_list = traceback.extract_tb(exc_traceback)

            # Find the frame that corresponds to our executed code
            error_line_num = None
            error_line_content = None

            for frame in tb_list:
                # The exec() call creates a frame with filename '<string>'
                if frame.filename == "<string>":
                    error_line_num = frame.lineno
                    if 1 <= error_line_num <= len(command_lines):
                        error_line_content = command_lines[error_line_num - 1].strip()
                    break

            # Create a detailed error message
            error_details = []
            error_details.append(f"Error Type: {exc_type.__name__}")
            error_details.append(f"Error Message: {str(exc_value)}")

            if error_line_num is not None:
                error_details.append(f"Error Line: {error_line_num}")
                if error_line_content:
                    error_details.append(f"Error Code: {error_line_content}")

            # Add full traceback for complex errors
            if len(tb_list) > 1:
                error_details.append("\nFull Traceback:")
                error_details.append(traceback.format_exc())

            output = "\n".join(error_details)
        finally:
            sys.stdout = old_stdout
        return output

    command = command.strip("```").strip()
    return execute_in_repl(command)


def read_function_source_code(function_name: str) -> str:
    """Read the source code of a function from any module path.

    Parameters
    ----------
        function_name (str): Fully qualified function name (e.g., 'bioagentos.tool.support_tools.write_python_code')

    Returns
    -------
        str: The source code of the function

    """
    import importlib
    import inspect

    # Split the function name into module path and function name
    parts = function_name.split(".")
    module_path = ".".join(parts[:-1])
    func_name = parts[-1]

    try:
        # Import the module
        module = importlib.import_module(module_path)

        # Get the function object from the module
        function = getattr(module, func_name)

        # Get the source code of the function
        source_code = inspect.getsource(function)

        return source_code
    except (ImportError, AttributeError) as e:
        return f"Error: Could not find function '{function_name}'. Details: {str(e)}"


def get_error_line_info(command: str) -> str:
    """
    Execute Python command and return detailed error information including exact line numbers.

    Parameters:
        command (str): Python code to execute

    Returns:
        str: Execution result or detailed error information with line numbers
    """
    # Store original stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    # Redirect output
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    try:
        # Clean the command
        clean_command = command.strip("```python").strip("```").strip()

        # Split into lines for error reporting
        lines = clean_command.split("\n")

        # Execute the code
        exec(clean_command, globals())

        # Get normal output
        output = stdout_capture.getvalue()
        if not output:
            output = "Code executed successfully (no output)"

    except Exception as e:
        # Get exception details
        exc_type, exc_value, exc_traceback = sys.exc_info()

        # Extract traceback information
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)

        # Find the error line in our code
        error_line = None
        error_content = None

        # Parse traceback to find line number
        for tb_line in tb_lines:
            if "line " in tb_line and "<string>" in tb_line:
                # Extract line number from traceback
                import re

                match = re.search(r"line (\d+)", tb_line)
                if match:
                    error_line = int(match.group(1))
                    if 1 <= error_line <= len(lines):
                        error_content = lines[error_line - 1].strip()
                    break

        # Format error message
        error_info = []
        error_info.append(f"❌ {exc_type.__name__}: {exc_value}")

        if error_line:
            error_info.append(f"📍 Error at line {error_line}")
            if error_content:
                error_info.append(f"🔍 Code: {error_content}")

        # Add context lines around error
        if error_line and len(lines) > 1:
            error_info.append("\n📋 Code context:")
            start_line = max(1, error_line - 2)
            end_line = min(len(lines), error_line + 2)

            for i in range(start_line, end_line + 1):
                line_content = lines[i - 1]
                marker = ">>> " if i == error_line else "    "
                error_info.append(f"{marker}{i:2d}: {line_content}")

        output = "\n".join(error_info)

    finally:
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return output


# def request_human_feedback(question, context, reason_for_uncertainty):
#     """
#     Request human feedback on a question.

#     Parameters:
#         question (str): The question that needs human feedback.
#         context (str): Context or details that help the human understand the situation.
#         reason_for_uncertainty (str): Explanation for why the LLM is uncertain about its answer.

#     Returns:
#         str: The feedback provided by the human.
#     """
#     print("Requesting human feedback...")
#     print(f"Question: {question}")
#     print(f"Context: {context}")
#     print(f"Reason for Uncertainty: {reason_for_uncertainty}")

#     # Capture human feedback
#     human_response = input("Please provide your feedback: ")

#     return human_response
