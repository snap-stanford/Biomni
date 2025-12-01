#!/usr/bin/env python3
"""
Benchmark runner using GNU parallel/xargs for stable parallel execution
"""
import os
import sys
import subprocess
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biomni.config import default_config

default_config.llm = "gemini-3-pro-preview"
# default_config.llm = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
# default_config.llm = "us.anthropic.claude-sonnet-4-20250514-v1:0"
default_config.commercial_mode = True
default_config.use_tool_retriever = True
default_config.path = "/home/jaechang/biomni_hits_test/"
default_config.timeout_seconds = 100


def check_parallel_available():
    """Check if GNU parallel is available"""
    try:
        subprocess.run(
            ["parallel", "--version"], capture_output=True, check=True, timeout=5
        )
        return True
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return False


def generate_commands(
    indices_to_run,
    dataset,
    llm,
    output_dir,
    skip_existing,
):
    """Generate all execution commands"""
    commands = []
    script_path = Path(__file__).parent / "benchmark_single_task.py"

    for i in indices_to_run:
        cmd = [
            sys.executable,
            str(script_path),
            "--index",
            str(i),
            "--dataset",
            dataset,
            "--llm",
            llm,
            "--output-dir",
            str(output_dir),
        ]
        if skip_existing:
            cmd.append("--skip-existing")

        commands.append(" ".join(cmd))

    return commands


def execute_parallel(commands, max_workers, commands_file):
    """Execute commands using GNU parallel or xargs"""
    use_parallel = check_parallel_available()

    if max_workers > 1:
        if use_parallel:
            print(f"⚡ Running with GNU parallel (jobs={max_workers})...\n")
            # Use --eta instead of --bar for batch environments (no /dev/tty)
            parallel_cmd = (
                f"parallel --jobs {max_workers} --eta --halt never < {commands_file}"
            )

            subprocess.run(parallel_cmd, shell=True)
        else:
            print(f"⚠️  GNU parallel not found, using xargs (jobs={max_workers})...\n")
            xargs_cmd = (
                f"cat {commands_file} | xargs -P {max_workers} -I {{}} bash -c '{{}}'"
            )

            subprocess.run(xargs_cmd, shell=True)
    else:
        print(f"🔄 Running sequentially...\n")
        # Sequential execution
        with open(commands_file, "r") as f:
            for line in f:
                cmd = line.strip()
                if cmd:
                    subprocess.run(cmd, shell=True)


def parse_indices(indices_str):
    """
    Parse indices string to a list of integers.
    Supports comma-separated values and ranges (e.g., "0,5,10-15,20")

    Args:
        indices_str: String containing indices (e.g., "0,5,10-15,20")

    Returns:
        List of integers
    """
    if not indices_str:
        return None

    indices = set()
    parts = indices_str.split(",")

    for part in parts:
        part = part.strip()
        if "-" in part:
            # Handle range (e.g., "10-15")
            start, end = part.split("-")
            indices.update(range(int(start), int(end) + 1))
        else:
            # Handle single index
            indices.add(int(part))

    return sorted(list(indices))


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run benchmark on specified datasets (using GNU parallel/xargs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all datasets (including all BiomniEval1 tasks)
  python benchmark.py
  
  # Run specific dataset (original tasks)
  python benchmark.py -d DbQA
  python benchmark.py -d SeqQA
  python benchmark.py -d HLE
  
  # Run multiple datasets at once
  python benchmark.py -d DbQA SeqQA HLE
  python benchmark.py -d gwas_causal_gene_opentargets gwas_causal_gene_pharmaprojects
  
  # Run specific BiomniEval1 task
  python benchmark.py -d gwas_causal_gene_opentargets
  python benchmark.py -d gwas_causal_gene_gwas_catalog
  python benchmark.py -d rare_disease_diagnosis
  python benchmark.py -d crispr_delivery
  
  # Run specific indices
  python benchmark.py -d DbQA -i "0,5,10"
  
  # Run range of indices
  python benchmark.py -d gwas_variant_prioritization -i "0-10"
  
  # Run mixed indices and ranges
  python benchmark.py -d SeqQA -i "0,5-10,15,20-25"
  
  # Run with custom folder name
  python benchmark.py -d patient_gene_detection -f tmp
  
  # Skip tests where ans_* already exists
  python benchmark.py -d DbQA -s
  
  # Run with specific LLM model
  python benchmark.py -d DbQA -l gemini-2.5-flash
  
  # Combine options (multiple datasets with indices)
  python benchmark.py -d screen_gene_retrieval patient_gene_detection -f tmp -s -i "0-10"
        """,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        nargs="+",
        choices=[
            "DbQA",
            "SeqQA",
            "HLE",
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
            "all",
        ],
        default=["all"],
        help="Dataset/task to run benchmark on (default: all). Can specify multiple datasets.",
    )
    parser.add_argument(
        "--indices",
        "-i",
        type=str,
        default=None,
        help='Specific question indices to run (e.g., "0,5,10" or "0-10" or "0,5-10,15")',
    )
    parser.add_argument(
        "--max-workers",
        "-n",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--folder",
        "-f",
        type=str,
        default=None,
        help="Folder name to save results (e.g., 'tmp' saves to results/tmp). If not provided, uses timestamp.",
    )
    parser.add_argument(
        "--skip-existing",
        "-s",
        action="store_true",
        help="Skip tests where ans_* file already exists",
    )
    parser.add_argument(
        "--llm",
        "-l",
        type=str,
        default="gemini-2.5-pro",
        help="LLM model to use for benchmarking (default: gemini-2.5-pro).",
    )

    args = parser.parse_args()

    # Determine which datasets to run
    if "all" in args.dataset:
        datasets = [
            "DbQA",
            "SeqQA",
            "HLE",
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
    else:
        datasets = args.dataset

    # Parse indices if provided
    selected_indices = parse_indices(args.indices)

    print(f"🚀 Running benchmark on datasets: {datasets}")
    print(f"🤖 Using LLM model: {args.llm}")
    if selected_indices:
        print(f"📊 Running only indices: {selected_indices}")
    else:
        print("📊 Running all indices")
    print(f"⚡ Using {args.max_workers} parallel workers")
    if args.skip_existing:
        print(f"⏭️  Skip existing: Enabled (will skip tests where ans_* exists)")
    print()

    # Create directory for outputs
    if args.folder:
        working_dir = Path(f"results/{args.folder}")
        print(f"📁 Using specified folder: {working_dir}")
    else:
        time_step = datetime.now().strftime("%Y%m%d_%H%M%S")
        working_dir = Path(f"results/{time_step}")
        print(f"📁 Using timestamp folder: {working_dir}")

    working_dir.mkdir(parents=True, exist_ok=True)

    # Store absolute path before changing directory
    output_dir = working_dir.absolute()

    # Pre-load BiomniEval1 tasks to determine instance counts
    from biomni.eval import BiomniEval1

    biomni_evaluator = BiomniEval1()
    task_counts = {}

    all_commands = []

    for dataset in datasets:
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(exist_ok=True)

        # Determine the range of indices for this dataset
        if dataset == "HLE":
            max_index = 52
        elif dataset in ["DbQA", "SeqQA"]:
            max_index = 60
        else:
            # For BiomniEval1 tasks, get the count from the evaluator
            if dataset not in task_counts:
                task_df = biomni_evaluator.get_instances_by_task(dataset, split="val")
                task_counts[dataset] = len(task_df)
                print(f"  Task {dataset}: {task_counts[dataset]} instances")
            max_index = task_counts[dataset]

        # Use selected indices or all indices
        if selected_indices:
            # Filter out indices that are out of range for this dataset
            indices_to_run = [i for i in selected_indices if i < max_index]
            if len(selected_indices) != len(indices_to_run):
                print(
                    f"  ⚠️  Warning: Some indices are out of range for {dataset} "
                    f"(max: {max_index-1}), skipping them."
                )
        else:
            indices_to_run = range(max_index)

        # Generate commands for this dataset
        commands = generate_commands(
            indices_to_run=indices_to_run,
            dataset=dataset,
            llm=args.llm,
            output_dir=output_dir,
            skip_existing=args.skip_existing,
        )
        all_commands.extend(commands)

    print(f"\n📝 Total tasks to run: {len(all_commands)}")
    print()

    # Save all commands to file
    commands_file = output_dir / "commands.txt"
    with open(commands_file, "w") as f:
        for cmd in all_commands:
            f.write(cmd + "\n")

    print(f"💾 Commands saved to: {commands_file}")
    print()

    # Execute in parallel
    start_time = time.time()
    execute_parallel(all_commands, args.max_workers, commands_file)
    total_time = time.time() - start_time

    print(f"\n✅ Benchmark completed in {total_time:.1f}s!")
    print(f"📂 Results saved to: {output_dir}")
    print()
    print("💡 To evaluate results, run:")
    print(f"   python evaluate.py {output_dir}")
    print()
