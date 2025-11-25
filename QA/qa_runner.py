#!/usr/bin/env python3
"""
HITS AI Agent QA Runner
AI agent를 실행하여 QA 태스크를 평가하는 CLI 도구
"""

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from biomni.config import default_config

default_config.llm = "gemini-3-pro-preview"
# default_config.llm = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
# default_config.llm = "us.anthropic.claude-sonnet-4-20250514-v1:0"
default_config.commercial_mode = True
default_config.use_tool_retriever = True
default_config.path = "/workdir_efs/jaechang/work2/biomni_hits_test/biomni_data"
default_config.timeout_seconds = 3600

# Biomni HITS 모듈 import를 위한 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from qa_core import (
    Evaluator,
    ImageComparator,
    QAManager,
    ReportGenerator,
)


def extract_solution_from_response(response: str) -> str:
    """
    AI agent 응답에서 <solution>...</solution> 태그 내용 추출

    Args:
        response: AI agent의 전체 응답

    Returns:
        solution 태그 내용 (태그가 없으면 전체 응답 반환)
    """
    # <solution>...</solution> 패턴 추출
    pattern = r"<solution>(.*?)</solution>"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()
    else:
        # solution 태그가 없으면 전체 응답 반환
        print(
            "Warning: <solution> tag not found in agent response. Using full response."
        )
        return response.strip()


def run_agent_on_question(
    agent, question: str, task_id: str, input_data_files: Optional[List[str]] = None
) -> tuple[str, float]:
    """
    AI agent에게 질문을 주고 답변 생성

    Args:
        agent: A1_HITS agent 인스턴스
        question: 질문 텍스트
        task_id: 태스크 ID
        input_data_files: input data 파일 리스트 (optional)

    Returns:
        (답변, 실행시간) 튜플
    """
    print(f"\n{'='*60}")
    print(f"Running AI Agent on Task: {task_id}")
    if input_data_files:
        print(f"Input Data: {', '.join(input_data_files)}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # agent 실행 (go 사용 - run2.py 스타일)
        full_response_parts = []
        print("\n🤖 Agent is thinking...\n")

        for idx, output in enumerate(agent.go(question)):
            print(f"==================== Step {idx} ====================")

            if idx == 0:
                # 첫 번째 출력은 system prompt
                print("System prompt loaded")
                continue

            # Handle structured content (list with images) - extract text only
            if isinstance(output, list):
                # Extract text parts from structured content
                text_parts = [
                    item["text"]
                    for item in output
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                if text_parts:
                    full_response_parts.extend(text_parts)
            elif isinstance(output, str):
                full_response_parts.append(output)

        # 전체 응답 조합
        full_response = "\n".join(full_response_parts)

        # solution 태그 추출
        answer = extract_solution_from_response(full_response)

        execution_time = time.time() - start_time

        print(f"\n✅ Agent completed in {execution_time:.2f}s")
        print(f"Answer length: {len(answer)} characters")

        return answer, execution_time

    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n❌ Agent failed after {execution_time:.2f}s: {e}")
        import traceback

        traceback.print_exc()
        return f"Error: {str(e)}", execution_time


def save_agent_output(
    task_id: str, question: str, answer: str, output_dir: Path
) -> Path:
    """
    AI agent의 출력을 저장

    Args:
        task_id: 태스크 ID
        question: 질문 텍스트
        answer: 답변 텍스트
        output_dir: 출력 디렉토리

    Returns:
        저장된 답변 파일 경로
    """
    task_output_dir = output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)

    # 질문 저장
    question_file = task_output_dir / "question.md"
    question_file.write_text(question, encoding="utf-8")

    # 답변 저장
    answer_file = task_output_dir / "generated_answer.md"
    answer_file.write_text(answer, encoding="utf-8")

    # 이미지가 있으면 복사 (추후 구현 필요 시)
    # TODO: agent가 생성한 이미지를 찾아서 task 폴더로 복사하는 로직

    print(f"📁 Saved output to: {task_output_dir}")

    return answer_file


def evaluate_task(
    task_id: str,
    question: str,
    ground_truth: str,
    generated_answer: str,
    ground_truth_task_dir: Path,
    generated_task_dir: Path,
    evaluator: Evaluator,
    image_comparator: ImageComparator,
    execution_time: float,
) -> tuple:
    """
    태스크 평가 수행

    Args:
        task_id: 태스크 ID
        question: 질문
        ground_truth: 정답
        generated_answer: 생성된 답변
        ground_truth_task_dir: 정답 태스크 디렉토리
        generated_task_dir: 생성된 태스크 디렉토리
        evaluator: Evaluator 인스턴스
        image_comparator: ImageComparator 인스턴스
        execution_time: 실행 시간

    Returns:
        (evaluation_result, image_evaluation) 튜플
    """
    print(f"\n📊 Evaluating task: {task_id}")

    # 텍스트 평가
    evaluation_result = evaluator.evaluate_answer(
        task_id, question, ground_truth, generated_answer
    )

    # 이미지 평가
    image_evaluation = image_comparator.evaluate_images(
        ground_truth_markdown=ground_truth,
        generated_markdown=generated_answer,
        ground_truth_task_dir=ground_truth_task_dir,
        generated_task_dir=generated_task_dir,
        compare_visually=True,
    )

    return evaluation_result, image_evaluation


def run_qa_pipeline(
    qa_manager: QAManager,
    agent,
    evaluator: Evaluator,
    image_comparator: ImageComparator,
    report_generator: ReportGenerator,
    output_base_dir: Path,
    task_ids: Optional[List[str]] = None,
    category: Optional[str] = None,
) -> List[Dict]:
    """
    전체 QA 파이프라인 실행

    Args:
        qa_manager: QAManager 인스턴스
        agent: AI agent 인스턴스
        evaluator: Evaluator 인스턴스
        image_comparator: ImageComparator 인스턴스
        report_generator: ReportGenerator 인스턴스
        output_base_dir: 결과 저장 기본 디렉토리
        task_ids: 실행할 태스크 ID 리스트 (None이면 전체)
        category: 카테고리 필터

    Returns:
        모든 평가 결과 리스트
    """
    # 실행할 태스크 선택
    if task_ids:
        tasks = [
            qa_manager.get_task(tid) for tid in task_ids if qa_manager.get_task(tid)
        ]
    else:
        tasks = qa_manager.list_tasks(category=category)

    if not tasks:
        print("❌ No tasks to run!")
        return []

    print(f"\n🚀 Running QA pipeline on {len(tasks)} task(s)")

    # 실행 ID 생성 (타임스탬프)
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_output_dir = output_base_dir / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output directory: {run_output_dir}")

    all_results = []

    for idx, task in enumerate(tasks, 1):
        print(f"\n{'#'*60}")
        print(f"Task {idx}/{len(tasks)}: {task.task_id}")
        print(f"{'#'*60}")

        try:
            # 1. AI Agent 실행 (input data 정보 전달)
            generated_answer, execution_time = run_agent_on_question(
                agent, task.question, task.task_id, task.input_data
            )

            # 2. 출력 저장
            task_output_dir = run_output_dir / task.task_id
            save_agent_output(
                task.task_id, task.question, generated_answer, run_output_dir
            )

            # 3. 평가 수행
            ground_truth_task_dir = task.task_path if task.task_path else Path(".")
            generated_task_dir = task_output_dir

            evaluation_result, image_evaluation = evaluate_task(
                task.task_id,
                task.question,
                task.answer,
                generated_answer,
                ground_truth_task_dir,
                generated_task_dir,
                evaluator,
                image_comparator,
                execution_time,
            )

            # 4. 개별 태스크 리포트 생성
            report_path = task_output_dir / "evaluation.json"
            report_generator.generate_task_report(
                task.task_id,
                evaluation_result,
                image_evaluation,
                execution_time,
                report_path,
            )

            # 5. 결과 요약 출력
            report_generator.print_task_summary(
                report_generator.load_task_report(report_path)
            )

            # 결과 수집
            all_results.append(report_generator.load_task_report(report_path))

        except Exception as e:
            print(f"❌ Error processing task {task.task_id}: {e}")
            import traceback

            traceback.print_exc()

    # 6. 종합 리포트 생성
    if all_results:
        summary_report_path = run_output_dir / "summary_report.md"
        report_generator.generate_summary_report(all_results, summary_report_path)

        print(f"\n{'='*60}")
        print(f"✅ QA Pipeline Completed!")
        print(f"{'='*60}")
        print(f"Total tasks: {len(all_results)}")
        print(
            f"Passed: {sum(1 for r in all_results if r['summary']['overall_passed'])}"
        )
        print(f"Results saved to: {run_output_dir}")

    return all_results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="HITS AI Agent QA Runner")

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
        "--evaluation-prompt",
        type=str,
        help="Path to custom evaluation prompt file",
    )

    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available tasks and exit",
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

    # AI Agent 초기화
    print("\n🤖 Initializing AI Agent...")
    try:
        from biomni.agent.a1_hits import A1_HITS

        agent = A1_HITS()
        print("✅ AI Agent initialized")
    except Exception as e:
        print(f"❌ Failed to initialize AI Agent: {e}")
        import traceback

        traceback.print_exc()
        return

    # Evaluator 초기화
    print("\n📊 Initializing Evaluator...")
    try:
        from biomni.llm import get_llm

        llm_client = get_llm(model=default_config.llm)
        evaluator = Evaluator(llm_client, pass_threshold=args.pass_threshold)
        print("✅ Evaluator initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Evaluator: {e}")
        import traceback

        traceback.print_exc()
        return

    # Image Comparator 초기화
    print("\n🖼️  Initializing Image Comparator...")
    image_comparator = ImageComparator(ssim_threshold=args.ssim_threshold)
    print("✅ Image Comparator initialized")

    # Report Generator 초기화
    print("\n📄 Initializing Report Generator...")
    report_generator = ReportGenerator()
    print("✅ Report Generator initialized")

    # QA 파이프라인 실행
    run_qa_pipeline(
        qa_manager=qa_manager,
        agent=agent,
        evaluator=evaluator,
        image_comparator=image_comparator,
        report_generator=report_generator,
        output_base_dir=output_dir,
        task_ids=args.tasks,
        category=args.category,
    )


if __name__ == "__main__":
    main()
