#!/usr/bin/env python3
"""
Parse log_*.txt files and generate ans_*.json files.
Extracts predicted answers from [ANSWER] or <solution> tags.
"""
import os
import re
import json
import sys
import argparse
from pathlib import Path
from glob import glob

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biomni.task.lab_bench import lab_bench
from biomni.task.hle import humanity_last_exam
from biomni.task.biomni_eval1_task import biomni_eval1_task


def load_benchmark(dataset: str):
    """Load the appropriate benchmark based on dataset name."""
    if dataset == "HLE":
        return humanity_last_exam(
            path="/workdir_efs/jaechang/work2/biomni_hits_test/biomni_data/biomni_data//benchmark/",
        )
    elif dataset in ["DbQA", "SeqQA"]:
        return lab_bench(
            path="/workdir_efs/jaechang/work2/biomni_hits_test/biomni_data/biomni_data/benchmark/",
            dataset=dataset,
        )
    else:
        # Use BiomniEval1 task loader for all other tasks
        return biomni_eval1_task(task_name=dataset, split="val")


def is_vanilla_mode(log_content: str) -> bool:
    """
    Check if the log is from vanilla (direct LLM) mode.
    Vanilla logs contain 'Using vanilla LLM:' at the beginning.
    """
    return "Using vanilla LLM:" in log_content


def extract_answer_from_log(log_content: str) -> str:
    """
    Extract predicted answer from log content.
    Uses the same parsing logic as benchmark_single_task.py:
    - Vanilla mode: [ANSWER] tags only, fallback to full response
    - Agent mode: [ANSWER] or <solution> tags
    """
    predicted_answer = ""

    if is_vanilla_mode(log_content):
        # For vanilla mode, extract answer from [ANSWER] tags only (not <solution> tags)
        answer_pattern = r"\[ANSWER\](.*?)\[/ANSWER\]"
        matches = list(re.finditer(answer_pattern, log_content, re.DOTALL))
        if matches:
            last_match = matches[-1]
            predicted_answer = last_match.group(1).strip()
        else:
            # If no [ANSWER] tag found, extract the Response part as fallback
            # The log format is: "Using vanilla LLM: ...\nQuestion: ...\n\nResponse:\n<content>"
            response_match = re.search(r"Response:\n(.*)$", log_content, re.DOTALL)
            if response_match:
                predicted_answer = response_match.group(1).strip()
            else:
                predicted_answer = log_content.strip()
    else:
        # For agent mode, extract answer from either [ANSWER] tags or <solution> tags
        patterns = [r"\[ANSWER\](.*?)\[/ANSWER\]", r"<solution>(.*?)</solution>"]
        for pattern in patterns:
            matches = list(re.finditer(pattern, log_content, re.DOTALL))
            if matches:
                last_match = matches[-1]
                predicted_answer = last_match.group(1).strip()
                break

    return predicted_answer


def parse_log_file(
    log_path: str, benchmark=None, dataset: str = None, llm: str = None
) -> dict:
    """
    Parse a single log file and return result data.
    """
    with open(log_path, "r", encoding="utf-8") as f:
        log_content = f.read()

    # Extract index from filename (log_123.txt -> 123)
    filename = os.path.basename(log_path)
    match = re.match(r"log_(\d+)\.txt", filename)
    if match:
        index = int(match.group(1))
    else:
        index = -1

    # Extract predicted answer
    predicted_answer = extract_answer_from_log(log_content)

    # Get question data from benchmark if available
    question = ""
    choices = None
    correct_answer = ""

    if benchmark is not None and index >= 0:
        try:
            qa = benchmark.get_example(index=index)
            question = qa.get("prompt", "")
            choices = qa.get("choices", None)
            correct_answer = qa.get("answer", "")
        except Exception as e:
            print(f"Warning: Could not get benchmark data for index {index}: {e}")

    # Create result data (same structure as benchmark_single_task.py)
    result_data = {
        "index": index,
        "dataset": dataset or "",
        "llm": llm or "",
        "question": question,
        "choices": choices,
        "correct_answer": correct_answer,
        "predicted_answer": predicted_answer,
        "full_output": log_content,
    }

    return result_data


def process_directory(input_dir: str, output_dir: str = None, overwrite: bool = False):
    """
    Process all log_*.txt files in the given directory and subdirectories.
    Generate corresponding ans_*.json files.

    Expects directory structure: {input_dir}/{dataset}/log_*.txt
    or: results/{llm}/{dataset}/log_*.txt
    """
    input_path = Path(input_dir)

    # Find all log_*.txt files
    log_files = list(input_path.rglob("log_*.txt"))

    if not log_files:
        print(f"No log_*.txt files found in {input_dir}")
        return

    print(f"Found {len(log_files)} log files to process")

    # Group log files by dataset (parent directory name)
    # and load benchmarks for each dataset
    datasets_benchmarks = {}

    processed = 0
    skipped = 0
    errors = 0

    for log_file in sorted(log_files):
        try:
            # Extract dataset from parent directory name
            dataset = log_file.parent.name

            # Extract llm from grandparent directory name (if exists)
            llm = ""
            if log_file.parent.parent != input_path:
                llm = log_file.parent.parent.name

            # Load benchmark for this dataset if not already loaded
            if dataset not in datasets_benchmarks:
                try:
                    print(f"Loading benchmark for dataset: {dataset}")
                    datasets_benchmarks[dataset] = load_benchmark(dataset)
                except Exception as e:
                    print(f"Warning: Could not load benchmark for {dataset}: {e}")
                    datasets_benchmarks[dataset] = None

            benchmark = datasets_benchmarks[dataset]

            # Determine output path
            if output_dir:
                # Preserve subdirectory structure
                rel_path = log_file.relative_to(input_path)
                ans_filename = re.sub(r"log_(\d+)\.txt", r"ans_\1.json", log_file.name)
                ans_file = Path(output_dir) / rel_path.parent / ans_filename
            else:
                # Same directory as log file
                ans_filename = re.sub(r"log_(\d+)\.txt", r"ans_\1.json", log_file.name)
                ans_file = log_file.parent / ans_filename

            # Skip if exists and not overwriting
            if ans_file.exists() and not overwrite:
                print(f"⏭ Skipping {ans_file} (already exists)")
                skipped += 1
                continue

            # Parse log and create ans file
            result_data = parse_log_file(
                str(log_file), benchmark=benchmark, dataset=dataset, llm=llm
            )

            # Ensure output directory exists
            ans_file.parent.mkdir(parents=True, exist_ok=True)

            # Write ans file
            with open(ans_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

            answer_preview = (
                result_data["predicted_answer"][:50] + "..."
                if len(result_data["predicted_answer"]) > 50
                else result_data["predicted_answer"]
            )
            print(
                f"✓ {log_file.name} -> {ans_file.name} (pred: {answer_preview}, correct: {result_data['correct_answer']})"
            )
            processed += 1

        except Exception as e:
            print(f"✗ Error processing {log_file}: {e}")
            errors += 1

    print(f"\nSummary: {processed} processed, {skipped} skipped, {errors} errors")


def main():
    parser = argparse.ArgumentParser(
        description="Parse log_*.txt files and generate ans_*.json files"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing log_*.txt files (recursive search)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: same as input)",
    )
    parser.add_argument(
        "--overwrite",
        "-f",
        action="store_true",
        help="Overwrite existing ans_*.json files",
    )

    args = parser.parse_args()

    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
