#!/usr/bin/env python3
"""
Single task runner for benchmark
Designed to be called by benchmark.py in parallel
"""
import os
import sys
import json
import re
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biomni.task.lab_bench import lab_bench
from biomni.task.hle import humanity_last_exam
from biomni.task.biomni_eval1_task import biomni_eval1_task
from biomni.agent import A1_HITS
from biomni.config import default_config
from biomni.llm import get_llm
from langchain_core.messages import HumanMessage


def convert_vanilla_model_name(llm_name: str) -> str:
    """
    Convert vanilla model name to actual model identifier.
    Removes '_vanilla' or '_vanila' suffix.
    e.g., 'gemini_pro_vanilla' -> 'gemini-pro'
    e.g., 'gemini-3-pro-preview_vanila' -> 'gemini-3-pro-preview'
    """
    if llm_name.endswith("_vanilla"):
        base_name = llm_name[:-8]  # Remove '_vanilla' suffix
        # Convert snake_case to kebab-case (only if there are underscores)
        if "_" in base_name:
            model_name = base_name.replace("_", "-")
        else:
            model_name = base_name
        return model_name
    elif llm_name.endswith("_vanila"):
        base_name = llm_name[:-7]  # Remove '_vanila' suffix
        # Convert snake_case to kebab-case (only if there are underscores)
        if "_" in base_name:
            model_name = base_name.replace("_", "-")
        else:
            model_name = base_name
        return model_name
    return llm_name


def run_single_task(
    index: int,
    dataset: str,
    llm: str,
    output_dir: str,
    skip_existing: bool = False,
):
    """Run a single benchmark task"""

    # Setup output files
    log_file = os.path.join(output_dir, dataset, f"log_{index}.txt")
    ans_file = os.path.join(output_dir, dataset, f"ans_{index}.json")

    # Skip if ans file exists and skip_existing is True
    if skip_existing and os.path.exists(ans_file):
        print(f"✓ Skipping {dataset}/ans_{index}.json (already exists)")
        return

    # Check if vanilla mode (direct LLM invocation)
    # Support both '_vanilla' and '_vanila' (typo)
    use_vanilla = llm.endswith("_vanilla") or llm.endswith("_vanila")

    # Determine model_id
    if llm == "kimi-k2-instruct":
        model_id = "accounts/fireworks/models/kimi-k2-instruct"
    elif use_vanilla:
        model_id = convert_vanilla_model_name(llm)
    else:
        model_id = llm

    try:
        # Load appropriate benchmark based on dataset
        if dataset == "HLE":
            benchmark = humanity_last_exam(
                path="/workdir_efs/jaechang/work2/biomni_hits_test/biomni_data/biomni_data//benchmark/",
            )
        elif dataset in ["DbQA", "SeqQA"]:
            benchmark = lab_bench(
                path="/workdir_efs/jaechang/work2/biomni_hits_test/biomni_data/biomni_data/benchmark/",
                dataset=dataset,
            )
        else:
            # Use BiomniEval1 task loader for all other tasks
            benchmark = biomni_eval1_task(task_name=dataset, split="val")

        qa = benchmark.get_example(index=index)

        # Use vanilla (direct LLM) or A1_HITS agent
        if use_vanilla:
            # Direct LLM invocation
            llm_instance = get_llm(model=model_id, config=default_config)
            prompt = qa["prompt"]

            # Run LLM and save logs
            with open(log_file, "w") as f:
                f.write(f"Using vanilla LLM: {model_id}\n")
                f.write(f"Question: {prompt}\n\n")
                f.flush()

                response = llm_instance.invoke([HumanMessage(content=prompt)])
                output = (
                    response.content if hasattr(response, "content") else str(response)
                )

                f.write(f"Response:\n{output}\n")
                f.flush()

            # For vanilla mode, extract answer from [ANSWER] tags only (not <solution> tags)
            predicted_answer = ""
            answer_pattern = r"\[ANSWER\](.*?)\[/ANSWER\]"
            matches = list(re.finditer(answer_pattern, output, re.DOTALL))
            if matches:
                last_match = matches[-1]
                predicted_answer = last_match.group(1).strip()
            else:
                # If no [ANSWER] tag found, use full response as fallback
                predicted_answer = output.strip()
        else:
            # Use A1_HITS agent
            agent = A1_HITS(llm=model_id)
            output = ""

            # Run agent and save logs
            with open(log_file, "w") as f:
                for idx, output_chunk in enumerate(agent.go(qa["prompt"])):
                    output += output_chunk
                    f.write(output_chunk + "\n")
                    f.write(str(agent.timer) + "\n")
                    f.flush()

            # Extract answer from either [ANSWER] tags or <solution> tags
            predicted_answer = ""
            patterns = [r"\[ANSWER\](.*?)\[/ANSWER\]", r"<solution>(.*?)</solution>"]
            for pattern in patterns:
                matches = list(re.finditer(pattern, output, re.DOTALL))
                if matches:
                    last_match = matches[-1]
                    predicted_answer = last_match.group(1).strip()
                    break

        # Create JSON structure with question, choices, predicted answer, and correct answer
        result_data = {
            "index": index,
            "dataset": dataset,
            "llm": llm,
            "question": qa.get("prompt", ""),
            "choices": qa.get("choices", None),
            "correct_answer": qa.get("answer", ""),
            "predicted_answer": predicted_answer,
            "full_output": output,
        }

        with open(ans_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        print(f"✓ Completed {dataset} index {index}")

    except Exception as e:
        print(f"✗ Error running {dataset} index {index}: {e}")
        # Save error result
        error_data = {
            "index": index,
            "dataset": dataset,
            "llm": llm,
            "error": str(e),
            "correct_answer": "",
            "predicted_answer": "",
        }
        with open(ans_file, "w", encoding="utf-8") as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)
        raise


def main():
    parser = argparse.ArgumentParser(description="Run a single benchmark task")
    parser.add_argument("--index", type=int, required=True, help="Task index")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--llm", type=str, required=True, help="LLM model name")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if output file exists",
    )

    args = parser.parse_args()

    run_single_task(
        index=args.index,
        dataset=args.dataset,
        llm=args.llm,
        output_dir=args.output_dir,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
