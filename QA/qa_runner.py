#!/usr/bin/env python3
"""
HITS AI Agent QA Runner (Parallel Wrapper)
qa_single_task.py를 parallel로 실행하는 wrapper 스크립트

Simple and clean architecture:
- qa_single_task.py: 단일 task 실행 (완전히 독립)
- qa_runner_simple.py: 전체 파이프라인 관리 및 병렬 실행
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Biomni HITS 모듈 import를 위한 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from biomni.config import default_config

# default_config 설정
default_config.llm = "gemini-3-pro-preview"
default_config.commercial_mode = True
default_config.use_tool_retriever = True
default_config.path = "/workdir_efs/jaechang/work2/biomni_hits_test/biomni_data"
default_config.timeout_seconds = 3600

from qa_core import QAManager, ReportGenerator


def check_parallel_available():
    """GNU parallel 사용 가능 여부 확인"""
    try:
        result = subprocess.run(
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
    tasks: List,
    num_repeats: int,
    qa_datasets_dir: Path,
    run_output_dir: Path,
    pass_threshold: float,
    ssim_threshold: float,
) -> List[str]:
    """모든 실행 커맨드 생성"""
    commands = []
    script_path = Path(__file__).parent / "qa_single_task.py"

    for task in tasks:
        for attempt_num in range(1, num_repeats + 1):
            cmd = [
                sys.executable,
                str(script_path),
                "--task-id",
                task.task_id,
                "--attempt",
                str(attempt_num),
                "--total-attempts",
                str(num_repeats),
                "--qa-datasets-dir",
                str(qa_datasets_dir),
                "--output-dir",
                str(run_output_dir),
                "--pass-threshold",
                str(pass_threshold),
                "--ssim-threshold",
                str(ssim_threshold),
            ]
            commands.append(" ".join(cmd))

    return commands


def execute_parallel(commands: List[str], max_workers: int, commands_file: Path):
    """GNU parallel 또는 xargs로 병렬 실행"""
    use_parallel = check_parallel_available()

    if max_workers > 1:
        if use_parallel:
            print(f"⚡ Running with GNU parallel (jobs={max_workers})...\n")
            parallel_cmd = (
                f"parallel --jobs {max_workers} --bar --halt never < {commands_file}"
            )

            subprocess.run(
                parallel_cmd,
                shell=True,
            )
        else:
            print(f"⚠️  GNU parallel not found, using xargs (jobs={max_workers})...\n")
            xargs_cmd = (
                f"cat {commands_file} | xargs -P {max_workers} -I {{}} bash -c '{{}}'"
            )

            subprocess.run(
                xargs_cmd,
                shell=True,
            )
    else:
        print(f"🔄 Running sequentially...\n")
        # 순차 실행
        with open(commands_file, "r") as f:
            for line in f:
                cmd = line.strip()
                if cmd:
                    subprocess.run(cmd, shell=True)


def collect_results(
    run_output_dir: Path,
    tasks: List,
    num_repeats: int,
) -> List[Dict]:
    """각 attempt의 결과를 수집하고 종합"""
    print("\n📊 Collecting results...")

    all_attempt_results = []
    total_runs = len(tasks) * num_repeats

    # 모든 evaluation.json 파일 수집
    for task in tasks:
        for attempt_num in range(1, num_repeats + 1):
            eval_file = (
                run_output_dir
                / task.task_id
                / f"attempt_{attempt_num}"
                / "evaluation.json"
            )

            if eval_file.exists():
                try:
                    with open(eval_file, "r", encoding="utf-8") as f:
                        result = json.load(f)
                        result["task_id"] = task.task_id
                        result["attempt_num"] = attempt_num
                        all_attempt_results.append(result)
                except Exception as e:
                    print(f"⚠️  Failed to load {eval_file}: {e}")
            else:
                print(f"⚠️  Missing result: {task.task_id} attempt {attempt_num}")
                all_attempt_results.append(
                    {
                        "task_id": task.task_id,
                        "attempt_num": attempt_num,
                        "summary": {
                            "overall_passed": False,
                            "error": "Result file not found",
                        },
                        "execution_time": 0,
                    }
                )

    print(f"✅ Collected {len(all_attempt_results)}/{total_runs} results")

    # 태스크별로 결과 집계
    task_final_results = {}

    for result in all_attempt_results:
        task_id = result.get("task_id")
        if task_id not in task_final_results:
            task_final_results[task_id] = {
                "task_id": task_id,
                "attempts": [],
                "all_passed": True,
            }

        task_final_results[task_id]["attempts"].append(result)
        if not result.get("summary", {}).get("overall_passed", False):
            task_final_results[task_id]["all_passed"] = False

    # 최종 결과 생성
    final_results = []
    for task_id, task_result in task_final_results.items():
        passed_count = sum(
            1
            for attempt in task_result["attempts"]
            if attempt.get("summary", {}).get("overall_passed", False)
        )
        total_count = len(task_result["attempts"])

        execution_times = [
            attempt.get("execution_time", 0) for attempt in task_result["attempts"]
        ]
        avg_execution_time = (
            sum(execution_times) / len(execution_times) if execution_times else 0
        )

        final_result = {
            "task_id": task_id,
            "num_attempts": total_count,
            "passed_attempts": passed_count,
            "all_attempts_passed": task_result["all_passed"],
            "avg_execution_time": avg_execution_time,
            "summary": {
                "overall_passed": task_result["all_passed"],
            },
            "attempts": task_result["attempts"],
        }
        final_results.append(final_result)

    return final_results


def print_summary(
    final_results: List[Dict],
    run_id: str,
    run_output_dir: Path,
    num_repeats: int,
    total_runs: int,
    max_workers: int,
    pass_threshold: float,
    ssim_threshold: float,
):
    """예쁜 통계 출력"""
    if not final_results:
        return

    total_tasks = len(final_results)
    passed_tasks = sum(1 for r in final_results if r["all_attempts_passed"])
    failed_tasks = total_tasks - passed_tasks
    success_rate = (passed_tasks / total_tasks * 100) if total_tasks > 0 else 0

    total_execution_time = sum(
        sum(attempt.get("execution_time", 0) for attempt in r["attempts"])
        for r in final_results
    )
    avg_time_per_task = total_execution_time / total_runs if total_runs > 0 else 0

    # 경로 축약 함수
    def truncate_path(path_str, max_len=60):
        """경로가 너무 길면 축약"""
        if len(path_str) <= max_len:
            return path_str
        parts = path_str.split("/")
        if len(parts) <= 2:
            return path_str
        return f".../{'/'.join(parts[-2:])}"

    output_path_display = truncate_path(str(run_output_dir), 60)

    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 27 + "QA PIPELINE SUMMARY" + " " * 32 + "║")
    print("╠" + "═" * 78 + "╣")
    print(f"║  📋 Run ID          : {run_id:<55} ║")
    print(f"║  📁 Output Path     : {output_path_display:<55} ║")
    print("╠" + "═" * 78 + "╣")

    # 실행 설정
    print("║" + " " * 31 + "CONFIGURATION" + " " * 34 + "║")
    print("╠" + "─" * 78 + "╣")
    print(
        f"║  Tasks              : {total_tasks:<8}   Repeats/Task  : {num_repeats:<8}   Total Runs  : {total_runs:<8} ║"
    )
    print(
        f"║  Max Workers        : {max_workers:<8}   Pass Threshold: {pass_threshold:.1f}%{' ' * 5}   SSIM Thresh.: {ssim_threshold:<8} ║"
    )
    print("╠" + "═" * 78 + "╣")

    # 결과 통계
    print("║" + " " * 34 + "RESULTS" + " " * 37 + "║")
    print("╠" + "─" * 78 + "╣")
    print(
        f"║  ✅ Passed          : {passed_tasks}/{total_tasks} tasks{' ' * (50 - len(str(passed_tasks)) - len(str(total_tasks)))}║"
    )
    print(
        f"║  ❌ Failed          : {failed_tasks}/{total_tasks} tasks{' ' * (50 - len(str(failed_tasks)) - len(str(total_tasks)))}║"
    )
    print(f"║  📊 Success Rate    : {success_rate:>6.1f}%{' ' * 51}║")
    print("╠" + "─" * 78 + "╣")
    print(f"║  ⏱️  Total Time      : {total_execution_time:>8.1f}s{' ' * 48}║")
    print(f"║  ⚡ Avg Time/Run    : {avg_time_per_task:>8.1f}s{' ' * 48}║")
    print("╠" + "═" * 78 + "╣")

    # 태스크별 상세 결과
    print("║" + " " * 30 + "TASK DETAILS" + " " * 36 + "║")
    print("╠" + "─" * 78 + "╣")

    for idx, result in enumerate(final_results, 1):
        task_id = result["task_id"]
        all_passed = result["all_attempts_passed"]
        passed_count = result["passed_attempts"]
        total_count = result["num_attempts"]
        avg_time = result["avg_execution_time"]

        status_icon = "✅" if all_passed else "❌"

        # 더 깔끔한 정렬
        task_name = f"{task_id:<20}"
        pass_info = f"{passed_count:>2}/{total_count:<2} passed"
        time_info = f"{avg_time:>7.1f}s avg"

        print(f"║  {status_icon} {task_name} │ {pass_info} │ {time_info}{' ' * 13}║")

    print("╚" + "═" * 78 + "╝")
    print()

    # 최종 상태 메시지
    if passed_tasks == total_tasks:
        print("\n" + "🎉" * 40)
        print("🎉" + "ALL TASKS PASSED!".center(78) + "🎉")
        print("🎉" * 40 + "\n")
    elif passed_tasks > 0:
        print("\n" + "⚠️ " * 20)
        print("⚠️ " + "SOME TASKS FAILED".center(76) + " ⚠️")
        print("⚠️ " * 20 + "\n")
    else:
        print("\n" + "❌" * 40)
        print("❌" + "ALL TASKS FAILED".center(78) + "❌")
        print("❌" * 40 + "\n")


def generate_report(
    final_results: List[Dict],
    run_id: str,
    run_output_dir: Path,
    num_repeats: int,
    total_runs: int,
    max_workers: int,
    pass_threshold: float,
    ssim_threshold: float,
):
    """마크다운 리포트 생성"""
    summary_report_path = run_output_dir / "summary_report.md"

    total_tasks = len(final_results)
    passed_tasks = sum(1 for r in final_results if r["all_attempts_passed"])
    failed_tasks = total_tasks - passed_tasks
    success_rate = (passed_tasks / total_tasks * 100) if total_tasks > 0 else 0

    total_execution_time = sum(
        sum(attempt.get("execution_time", 0) for attempt in r["attempts"])
        for r in final_results
    )
    avg_time_per_task = total_execution_time / total_runs if total_runs > 0 else 0

    with open(summary_report_path, "w", encoding="utf-8") as f:
        f.write(f"# QA Pipeline Summary Report\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- **Run ID**: {run_id}\n")
        f.write(f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Total Tasks**: {len(final_results)}\n")
        f.write(f"- **Repeats per Task**: {num_repeats}\n")
        f.write(f"- **Total Runs**: {total_runs}\n")
        f.write(f"- **Max Workers**: {max_workers}\n")
        f.write(f"- **Pass Threshold**: {pass_threshold}%\n")
        f.write(f"- **SSIM Threshold**: {ssim_threshold}\n\n")

        f.write(f"## Overall Results\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| ✅ Passed Tasks | {passed_tasks}/{total_tasks} |\n")
        f.write(f"| ❌ Failed Tasks | {failed_tasks}/{total_tasks} |\n")
        f.write(f"| 📊 Success Rate | {success_rate:.1f}% |\n")
        f.write(f"| ⏱️  Total Execution Time | {total_execution_time:.1f}s |\n")
        f.write(f"| ⚡ Avg Time per Run | {avg_time_per_task:.1f}s |\n\n")

        f.write(f"## Task Details\n\n")
        f.write(f"| Status | Task ID | Attempts Passed | Avg Time |\n")
        f.write(f"|--------|---------|----------------|----------|\n")

        for result in final_results:
            task_id = result["task_id"]
            all_passed = result["all_attempts_passed"]
            passed_count = result["passed_attempts"]
            total_count = result["num_attempts"]
            avg_time = result["avg_execution_time"]

            status = "✅ PASS" if all_passed else "❌ FAIL"
            f.write(
                f"| {status} | {task_id} | {passed_count}/{total_count} | {avg_time:.1f}s |\n"
            )

        f.write(f"\n## Individual Attempts\n\n")
        for result in final_results:
            task_id = result["task_id"]
            all_passed = result["all_attempts_passed"]

            status = "✅ PASS" if all_passed else "❌ FAIL"
            f.write(f"### {task_id}: {status}\n\n")
            f.write(f"**Summary**: All {result['num_attempts']} attempts must pass. ")
            f.write(
                f"Result: {result['passed_attempts']}/{result['num_attempts']} passed.\n\n"
            )

            for idx, attempt in enumerate(result["attempts"], 1):
                attempt_passed = attempt.get("summary", {}).get("overall_passed", False)
                attempt_time = attempt.get("execution_time", 0)
                attempt_icon = "✅" if attempt_passed else "❌"
                f.write(
                    f"- {attempt_icon} Attempt {idx}: {'PASSED' if attempt_passed else 'FAILED'} ({attempt_time:.1f}s)\n"
                )

            f.write("\n")

    print(f"📄 Detailed report saved: {summary_report_path}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="HITS AI Agent QA Runner (Parallel Wrapper)"
    )

    parser.add_argument(
        "--qa-datasets-dir",
        type=str,
        default="qa_datasets",
        help="QA datasets directory (default: qa_datasets)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="qa_results",
        help="Output directory for results (default: qa_results)",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        help="Specific task IDs to run (default: all tasks)",
    )

    parser.add_argument(
        "--category",
        type=str,
        help="Filter tasks by category",
    )

    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=70.0,
        help="Pass threshold score (0-100, default: 70)",
    )

    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.8,
        help="SSIM threshold for image comparison (0-1, default: 0.8)",
    )

    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available tasks and exit",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat each task (default: 1)",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of parallel workers (default: 1)",
    )

    args = parser.parse_args()

    # 경로 설정
    script_dir = Path(__file__).parent
    qa_datasets_dir = script_dir / args.qa_datasets_dir
    output_dir = script_dir / args.output_dir

    # QA Manager 초기화
    print("📦 Initializing QA Manager...")
    qa_manager = QAManager(qa_datasets_dir)

    if args.list_tasks:
        print("\n📋 Available Tasks:")
        print(f"{'='*60}")
        for task in qa_manager.list_tasks():
            print(
                f"  - {task.task_id} (Category: {task.category}, Difficulty: {task.difficulty})"
            )
        print(f"{'='*60}")
        print(f"Total: {qa_manager.get_task_count()} tasks")
        return

    if qa_manager.get_task_count() == 0:
        print("❌ No tasks found! Please add tasks to the qa_datasets directory.")
        return

    # 실행할 태스크 선택
    if args.tasks:
        tasks = [
            qa_manager.get_task(tid) for tid in args.tasks if qa_manager.get_task(tid)
        ]
    else:
        tasks = qa_manager.list_tasks(category=args.category)

    if not tasks:
        print("❌ No tasks to run!")
        return

    total_runs = len(tasks) * args.repeat

    print(f"\n🚀 Running QA pipeline")
    print(f"  - Tasks: {len(tasks)}")
    print(f"  - Repeats per task: {args.repeat}")
    print(f"  - Total runs: {total_runs}")
    print(f"  - Max parallel workers: {args.max_workers}")

    # 실행 ID 생성
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_output_dir = output_dir / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output directory: {run_output_dir}")

    # 커맨드 생성
    print("\n📝 Generating commands...")
    commands = generate_commands(
        tasks=tasks,
        num_repeats=args.repeat,
        qa_datasets_dir=qa_datasets_dir,
        run_output_dir=run_output_dir,
        pass_threshold=args.pass_threshold,
        ssim_threshold=args.ssim_threshold,
    )

    commands_file = run_output_dir / "commands.txt"
    with open(commands_file, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")

    print(f"✅ Generated {len(commands)} commands")
    print(f"Commands saved to: {commands_file}")

    # 병렬 실행
    start_time = time.time()
    execute_parallel(commands, args.max_workers, commands_file)
    total_time = time.time() - start_time

    print(f"\n✅ All tasks completed in {total_time:.1f}s")

    # 결과 수집
    final_results = collect_results(run_output_dir, tasks, args.repeat)

    # 통계 출력
    print_summary(
        final_results=final_results,
        run_id=run_id,
        run_output_dir=run_output_dir,
        num_repeats=args.repeat,
        total_runs=total_runs,
        max_workers=args.max_workers,
        pass_threshold=args.pass_threshold,
        ssim_threshold=args.ssim_threshold,
    )

    # 리포트 생성
    generate_report(
        final_results=final_results,
        run_id=run_id,
        run_output_dir=run_output_dir,
        num_repeats=args.repeat,
        total_runs=total_runs,
        max_workers=args.max_workers,
        pass_threshold=args.pass_threshold,
        ssim_threshold=args.ssim_threshold,
    )


if __name__ == "__main__":
    main()
