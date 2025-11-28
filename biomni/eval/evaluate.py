import glob
import os
import argparse
import json
import sys

# Add Biomni_HITS to path to import BiomniEval1
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Biomni_HITS"))

try:
    from biomni.eval import BiomniEval1

    BIOMNI_EVAL1_AVAILABLE = True
except ImportError:
    BIOMNI_EVAL1_AVAILABLE = False
    print("Warning: BiomniEval1 not available. Using simple string comparison only.")


# List of BiomniEval1 task names
BIOMNI_EVAL1_TASKS = [
    "crispr_delivery",
    "gwas_causal_gene_opentargets",
    "gwas_causal_gene_pharmaprojects",
    "gwas_causal_gene_gwas_catalog",
    "gwas_variant_prioritization",
    "lab_bench_dbqa",
    "lab_bench_seqqa",
    "rare_disease_diagnosis",
    "screen_gene_retrieval",
    "patient_gene_detection",
]


def is_biomni_eval1_task(dataset_name):
    """Check if the dataset is a BiomniEval1 task"""
    return dataset_name in BIOMNI_EVAL1_TASKS


def evaluate_answer(dataset_name, correct_answer, predicted_answer, evaluator=None):
    """
    Evaluate a single answer using appropriate logic for the dataset.

    Args:
        dataset_name: Name of the dataset/task
        correct_answer: Ground truth answer
        predicted_answer: Predicted answer
        evaluator: BiomniEval1 evaluator instance (if available)

    Returns:
        bool: True if correct, False otherwise
    """
    # For BiomniEval1 tasks, use the task-specific evaluation logic
    if is_biomni_eval1_task(dataset_name) and evaluator is not None:
        try:
            # BiomniEval1's _compute_reward returns 1.0 for correct, 0.0 for incorrect
            score = evaluator._compute_reward(
                dataset_name, predicted_answer, correct_answer
            )
            return score == 1.0
        except Exception as e:
            print(f"Warning: BiomniEval1 evaluation failed: {e}")
            # Fall back to simple comparison
            return correct_answer == predicted_answer
    else:
        # For original benchmarks (DbQA, SeqQA, HLE), use simple string comparison
        return correct_answer == predicted_answer


def evaluate_directory(base_dir, show_errors=False, verbose=False):
    """
    Evaluate benchmark results in the specified directory.

    Args:
        base_dir: Base directory containing dataset folders
        show_errors: If True, display indices of incorrect predictions
        verbose: If True, show detailed information about each error
    """
    # Find all dataset directories
    datasets = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not datasets:
        print(f"No dataset directories found in {base_dir}")
        return

    print(f"Found datasets: {datasets}")
    print()

    # Initialize BiomniEval1 evaluator if available
    evaluator = None
    if BIOMNI_EVAL1_AVAILABLE:
        try:
            evaluator = BiomniEval1()
            print("BiomniEval1 evaluator loaded successfully")
            print()
        except Exception as e:
            print(f"Warning: Could not load BiomniEval1 evaluator: {e}")
            print("Will use simple string comparison for all tasks")
            print()

    for dataset in sorted(datasets):
        dataset_path = os.path.join(base_dir, dataset)

        # Find all answer files directly in the dataset directory
        json_files = sorted(glob.glob(os.path.join(dataset_path, "ans_*.json")))
        txt_files = sorted(glob.glob(os.path.join(dataset_path, "ans_*.txt")))

        # Prioritize JSON files if available
        ans_files = json_files if json_files else txt_files
        is_json = bool(json_files)

        if not ans_files:
            continue

        # Group files by LLM model (reading from JSON content)
        llm_files = {}
        for ans_file in ans_files:
            filename = os.path.basename(ans_file)
            # Extract index from filename (e.g., ans_0.json -> 0)
            idx_str = (
                filename.replace("ans_", "").replace(".json", "").replace(".txt", "")
            )
            try:
                idx = int(idx_str)
            except ValueError:
                # Skip files that don't match the expected pattern
                continue

            if is_json:
                # Read LLM from JSON content
                try:
                    with open(ans_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        llm = data.get("llm", "unknown")
                except (json.JSONDecodeError, IOError):
                    llm = "unknown"
            else:
                # For old TXT format, assume default LLM
                llm = "unknown"

            if llm not in llm_files:
                llm_files[llm] = []
            llm_files[llm].append((idx, ans_file))

        for llm in sorted(llm_files.keys()):
            true, pred, file_indices = [], [], []
            error_details = []  # Store detailed error information
            no_answer_details = []  # Store no answer cases

            for idx, ans_file in sorted(llm_files[llm]):
                if is_json:
                    # Read JSON format
                    with open(ans_file, "r", encoding="utf-8") as f:
                        try:
                            data = json.load(f)
                            correct = data.get("correct_answer", "")
                            predicted = data.get("predicted_answer", "")

                            true.append(correct)
                            pred.append(predicted)
                            file_indices.append(idx)

                            # Store error details if incorrect or no answer
                            if predicted == "":
                                no_answer_details.append(
                                    {
                                        "index": idx,
                                        "question": data.get("question", ""),
                                        "correct": correct,
                                        "predicted": predicted,
                                        "file": ans_file,
                                    }
                                )
                            elif correct != predicted:  # Preliminary check
                                error_details.append(
                                    {
                                        "index": idx,
                                        "question": data.get("question", ""),
                                        "correct": correct,
                                        "predicted": predicted,
                                        "file": ans_file,
                                    }
                                )
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse JSON file {ans_file}")
                            continue
                else:
                    # Read old TXT format
                    with open(ans_file, "r") as f:
                        values = f.read().split("\n")
                        if len(values) >= 2:
                            true.append(values[0])
                            pred.append(values[1])
                            file_indices.append(idx)

            # Find incorrect predictions using appropriate evaluation logic
            incorrect_indices = []
            no_answer_indices = []
            correct_count = 0

            for i, (t, p, idx) in enumerate(zip(true, pred, file_indices)):
                if p == "":
                    no_answer_indices.append(idx)
                elif evaluate_answer(dataset, t, p, evaluator):
                    correct_count += 1
                else:
                    incorrect_indices.append(idx)

            incorrect_indices = sorted(incorrect_indices)
            no_answer_indices = sorted(no_answer_indices)

            # Filter error_details to only include actual errors (using proper evaluation logic)
            filtered_error_details = []
            for error in error_details:
                if not evaluate_answer(
                    dataset, error["correct"], error["predicted"], evaluator
                ):
                    filtered_error_details.append(error)
            error_details = filtered_error_details

            if len(true) == 0:
                continue

            print("=" * 30)
            print(f"{dataset} / {llm}")

            # Indicate if using BiomniEval1 task-specific evaluation
            if is_biomni_eval1_task(dataset) and evaluator is not None:
                print(f"[BiomniEval1 Task: using task-specific evaluation logic]")

            print(f"Number of correct predictions: {correct_count}")
            print(f"Number of no answer: {pred.count('')}")
            print(f"Total predictions: {len(true)}")
            print(f"Accuracy: {correct_count / len(true):.2%}")

            if show_errors:
                if incorrect_indices:
                    print(f"Incorrect predictions (indices): {incorrect_indices}")
                if no_answer_indices:
                    print(f"No answer (indices): {no_answer_indices}")

            # Show detailed error information only if verbose is enabled
            if verbose and show_errors:
                # Show detailed error information for JSON format
                if is_json and error_details:
                    print(
                        "\n--- Detailed Error Information (Incorrect Predictions) ---"
                    )
                    print(f"Total errors: {len(error_details)}\n")

                    # Show all errors
                    for error in error_details:
                        print(f"\nIndex {error['index']}:")
                        print(f"  Correct: {error['correct']}")
                        print(f"  Predicted: {error['predicted']}")

                        # Show question
                        preview_length = 500
                        q_preview = error["question"][:preview_length].replace(
                            "\n", " "
                        )
                        if len(error["question"]) > preview_length:
                            q_preview += "..."
                        print(f"  Question: {q_preview}")

                # Show detailed information for no answer cases
                if is_json and no_answer_details:
                    print("\n--- Detailed Error Information (No Answer Cases) ---")
                    print(f"Total no answer cases: {len(no_answer_details)}\n")

                    # Show all no answer cases
                    for no_ans in no_answer_details:
                        print(f"\nIndex {no_ans['index']}:")
                        print(f"  Correct: {no_ans['correct']}")
                        print(f"  Predicted: (no answer)")

                        # Show question
                        preview_length = 500
                        q_preview = no_ans["question"][:preview_length].replace(
                            "\n", " "
                        )
                        if len(no_ans["question"]) > preview_length:
                            q_preview += "..."
                        print(f"  Question: {q_preview}")

            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate benchmark results in a specified directory. "
        "Supports both original benchmarks (DbQA, SeqQA, HLE) and BiomniEval1 tasks "
        "with task-specific evaluation logic."
    )
    parser.add_argument(
        "directory",
        type=str,
        nargs="?",
        default=".",
        help="Directory containing benchmark results (default: current directory)",
    )
    parser.add_argument(
        "--show-errors",
        "-e",
        action="store_true",
        help="Show indices of incorrect predictions and no-answer cases",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information (correct/predicted answers and questions) for each error (requires --show-errors and JSON format)",
    )

    args = parser.parse_args()

    # Convert to absolute path
    eval_dir = os.path.abspath(args.directory)

    if not os.path.exists(eval_dir):
        print(f"Error: Directory '{eval_dir}' does not exist")
        exit(1)

    if not os.path.isdir(eval_dir):
        print(f"Error: '{eval_dir}' is not a directory")
        exit(1)

    print(f"Evaluating results in: {eval_dir}")
    print()
    evaluate_directory(eval_dir, show_errors=args.show_errors, verbose=args.verbose)
