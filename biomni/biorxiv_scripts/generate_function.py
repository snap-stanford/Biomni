#!/usr/bin/env python3
"""Command-line tool to generate Python functions from task descriptions using the function_generator agent."""

import argparse
import json
import os

from biomni.agent.function_generator import FunctionGenerator
from tqdm import tqdm


def main():
    """Main function for the command-line tool."""
    parser = argparse.ArgumentParser(description="Generate Python functions given task descriptions")
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        help="JSON file containing a list of task descriptions",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="generated_functions",
        help="Directory to save the generated functions (default: generated_functions)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="claude-3-7-sonnet-latest",
        help="LLM model to use (default: claude-3-7-sonnet-latest)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature setting for the LLM (default: 0.7)",
    )

    args = parser.parse_args()

    with open(args.task) as f_tasks:
        task_list = json.load(f_tasks)
        task_descriptions = list(task_list["tasks"])

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the function generator agent
    function_generator = FunctionGenerator(llm=args.model, temperature=args.temperature)

    if not task_list:
        print("No tasks found.")
    else:
        # tqdm shows a progress bar, file names as description
        for _i, desc in enumerate(tqdm(task_descriptions, desc="Generating Python scripts given task descriptions"), 1):
            generated_script_name, generated_codes = function_generator.go(desc)

            # Save results
            result_path = os.path.join(args.output_dir, generated_script_name)

            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path, "w") as f:
                f.write(generated_codes)

        print("DONE")


if __name__ == "__main__":
    main()
