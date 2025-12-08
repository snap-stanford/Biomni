#!/usr/bin/env python3
"""
HITS AI Agent QA Single Task Runner
단일 task/attempt를 독립적으로 실행하는 스크립트
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Biomni HITS 모듈 import를 위한 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from biomni.config import default_config
from biomni.agent import A1_HITS
from biomni.llm import get_llm

# default_config 설정
default_config.llm = "gemini-3-pro-preview"
default_config.commercial_mode = True
default_config.use_tool_retriever = True
default_config.path = "/workdir_efs/jaechang/work2/biomni_hits_test/biomni_data"
default_config.timeout_seconds = 3600

from qa_core import QAManager, Evaluator, ImageComparator


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
        solution_content = match.group(1).strip()
        print(f"✅ Extracted solution content ({len(solution_content)} chars)")
        return solution_content
    else:
        # solution 태그가 없으면 전체 응답 반환
        print("⚠️  <solution> tag not found. Using full response.")
        return response.strip()


def run_single_task(
    task_id: str,
    attempt_num: int,
    total_attempts: int,
    qa_datasets_dir: Path,
    output_dir: Path,
    pass_threshold: float,
    ssim_threshold: float,
) -> Dict[str, Any]:
    """
    단일 태스크 실행 및 평가

    Args:
        task_id: 태스크 ID
        attempt_num: 현재 시도 번호
        total_attempts: 총 시도 횟수
        qa_datasets_dir: QA 데이터셋 디렉토리
        output_dir: 출력 디렉토리
        pass_threshold: 통과 기준 점수
        ssim_threshold: SSIM 임계값

    Returns:
        평가 결과 딕셔너리
    """
    print(f"\n{'='*80}")
    print(f"🚀 Starting Task: {task_id} (Attempt {attempt_num}/{total_attempts})")
    print(f"{'='*80}\n")

    start_time = time.time()

    # 1. QA Manager로 태스크 로드
    print("📦 Loading task...")
    qa_manager = QAManager(qa_datasets_dir)
    task = qa_manager.get_task(task_id)

    if task is None:
        error_msg = f"Task {task_id} not found!"
        print(f"❌ {error_msg}")
        return {
            "task_id": task_id,
            "attempt_num": attempt_num,
            "summary": {
                "overall_passed": False,
                "error": error_msg,
            },
            "execution_time": 0,
        }

    print(f"✅ Task loaded: {task.task_id}")
    print(f"   Category: {task.category}")
    print(f"   Difficulty: {task.difficulty}")
    print(f"   Images: {len(task.images)}")
    print(f"   Input Data: {len(task.input_data)}")

    # 2. 출력 디렉토리 설정
    task_output_dir = output_dir / task_id / f"attempt_{attempt_num}"
    task_output_dir.mkdir(parents=True, exist_ok=True)

    # 2.1. Input data 파일들을 작업 디렉토리에 복사
    if task.input_data:
        print(f"\n📥 Copying input data files...")
        for data_file in task.input_data:
            src_path = task.task_path / data_file
            dst_path = task_output_dir / data_file
            if src_path.exists():
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"   ✓ Copied: {data_file}")
            else:
                print(f"   ⚠️  Not found: {data_file}")

    # 3. HITS 에이전트 초기화
    print("\n🤖 Initializing HITS Agent...")
    try:
        agent = A1_HITS()
        print("✅ Agent initialized")
    except Exception as e:
        error_msg = f"Failed to initialize agent: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback

        traceback.print_exc()
        return {
            "task_id": task_id,
            "attempt_num": attempt_num,
            "summary": {
                "overall_passed": False,
                "error": error_msg,
            },
            "execution_time": time.time() - start_time,
        }

    # 4. 에이전트 실행 (작업 디렉토리를 task_output_dir로 변경)
    print(f"\n🔄 Running agent with question...")
    print(f"Question preview: {task.question[:200]}...")

    # 현재 디렉토리 저장 및 작업 디렉토리 변경
    original_dir = os.getcwd()
    os.chdir(task_output_dir)
    print(f"📂 Changed working directory to: {task_output_dir}")

    try:
        # 에이전트에게 질문 전달 (generator 반환)
        agent_start_time = time.time()

        # agent.go()는 generator를 반환하므로 iterate해야 함
        full_response_parts = []
        system_prompt = None
        logs_content = []  # Step별 로그 저장

        print("🤖 Agent is thinking...\n")

        for idx, output in enumerate(agent.go(task.question)):
            print(f"==================== Step {idx} ====================")

            if idx == 0:
                # 첫 번째 출력은 system prompt
                system_prompt = output
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
                    text_content = "\n".join(text_parts)
                    full_response_parts.extend(text_parts)
                    logs_content.append(text_content)
                    logs_content.append(
                        f"===================={idx}===================="
                    )
            elif isinstance(output, str):
                full_response_parts.append(output)
                logs_content.append(output)
                logs_content.append(f"===================={idx}====================")

        # 전체 응답 조합
        full_response = "\n".join(full_response_parts)
        agent_execution_time = time.time() - agent_start_time

        print(f"\n✅ Agent completed in {agent_execution_time:.1f}s")
        print(f"   Total steps: {idx}")

        # Agent timer 정보 출력
        if hasattr(agent, "timer"):
            print(f"   Agent timer: {agent.timer}")

        # <solution> 태그에서 최종 답변 추출
        final_answer = extract_solution_from_response(full_response)

        print(f"   Full response length: {len(full_response)} characters")
        print(f"   Final answer length: {len(final_answer)} characters")
        print(f"\nFinal answer preview: {final_answer[:200]}...")

        # 전체 응답 저장 (full_response.md)
        full_response_file = task_output_dir / "full_response.md"
        with open(full_response_file, "w", encoding="utf-8") as f:
            f.write(full_response)
        print(f"📄 Full response saved to: {full_response_file}")

        # 최종 답변 저장 (agent_answer.md) - solution 태그 추출된 내용
        answer_file = task_output_dir / "agent_answer.md"
        with open(answer_file, "w", encoding="utf-8") as f:
            f.write(final_answer)
        print(f"📄 Final answer saved to: {answer_file}")

        # System prompt도 저장 (디버깅용)
        if system_prompt:
            system_prompt_file = task_output_dir / "system_prompt.txt"
            with open(system_prompt_file, "w", encoding="utf-8") as f:
                if isinstance(system_prompt, str):
                    f.write(system_prompt)
                else:
                    f.write(str(system_prompt))
            print(f"📄 System prompt saved to: {system_prompt_file}")

        # Step별 로그 저장 (디버깅용)
        if logs_content:
            logs_file = task_output_dir / "logs.txt"
            with open(logs_file, "w", encoding="utf-8") as f:
                f.write("\n".join(logs_content))
            print(f"📄 Step logs saved to: {logs_file}")

        # 평가에는 최종 답변 사용
        answer = final_answer

    except Exception as e:
        error_msg = f"Agent execution failed: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback

        traceback.print_exc()
        return {
            "task_id": task_id,
            "attempt_num": attempt_num,
            "summary": {
                "overall_passed": False,
                "error": error_msg,
            },
            "execution_time": time.time() - start_time,
        }
    finally:
        # 원래 디렉토리로 복귀
        os.chdir(original_dir)
        print(f"📂 Restored working directory to: {original_dir}")

    # 5. LLM 기반 답변 평가
    print(f"\n📊 Evaluating answer with LLM...")
    try:
        llm_client = get_llm(model="gemini-3-pro-preview")
        evaluation_prompt_path = (
            Path(__file__).parent / "qa_config" / "evaluation_prompt.txt"
        )
        evaluator = Evaluator(
            llm_client=llm_client,
            pass_threshold=pass_threshold,
            evaluation_prompt_path=(
                str(evaluation_prompt_path) if evaluation_prompt_path.exists() else None
            ),
        )

        eval_result = evaluator.evaluate_answer(
            task_id=task_id,
            question=task.question,
            ground_truth=task.answer,
            generated_answer=answer,
        )

        print(f"✅ LLM Evaluation:")
        print(f"   Content Accuracy: {eval_result.scores['content_accuracy']:.1f}")
        print(f"   Completeness: {eval_result.scores['completeness']:.1f}")
        print(f"   Format Compliance: {eval_result.scores['format_compliance']:.1f}")
        print(f"   Overall Score: {eval_result.overall_score:.1f}")
        print(f"   Passed: {'✅' if eval_result.passed else '❌'}")
        print(f"   Feedback: {eval_result.llm_feedback}")

    except Exception as e:
        error_msg = f"LLM evaluation failed: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback

        traceback.print_exc()
        # Continue with dummy evaluation
        eval_result = None

    # 6. 이미지 비교 (있는 경우)
    image_eval_result = None
    if task.images:
        print(f"\n🖼️  Evaluating images...")
        try:
            # Vision-capable LLM 사용
            vision_llm = get_llm(model="gemini-3-pro-preview")

            image_comparator = ImageComparator(
                ssim_threshold=ssim_threshold,
                llm_client=vision_llm,
                llm_threshold=pass_threshold,
            )

            # 이미지 평가 수행
            image_eval_result = image_comparator.evaluate_images(
                ground_truth_markdown=task.answer,
                generated_markdown=answer,
                ground_truth_task_dir=task.task_path,
                generated_task_dir=task_output_dir,
                question=task.question,
                compare_visually=True,
                use_llm_comparison=True,
            )

            print(f"✅ Image Evaluation:")
            print(f"   Expected: {len(image_eval_result.expected_images)} images")
            print(f"   Found: {len(image_eval_result.found_images)} images")
            print(f"   Missing: {len(image_eval_result.missing_images)} images")
            print(
                f"   All Present: {'✅' if image_eval_result.all_images_present else '❌'}"
            )

            if image_eval_result.average_similarity is not None:
                print(f"   Avg SSIM: {image_eval_result.average_similarity:.3f}")

            if image_eval_result.average_llm_score is not None:
                print(f"   LLM Image Score: {image_eval_result.average_llm_score:.1f}")

        except Exception as e:
            error_msg = f"Image evaluation failed: {str(e)}"
            print(f"⚠️  {error_msg}")
            import traceback

            traceback.print_exc()

    # 7. 최종 평가 결과 생성
    execution_time = time.time() - start_time

    # 통과 조건 판단
    text_passed = eval_result.passed if eval_result else False

    # 이미지가 있는 경우: 텍스트 + 이미지 모두 통과해야 함
    # 이미지가 없는 경우: 텍스트만 통과하면 됨
    if task.images:
        # 이미지 평가 통과 조건
        images_passed = False
        if image_eval_result:
            # 1. 모든 이미지가 존재하거나
            # 2. LLM이 내용이 모두 포함되어 있다고 판단하거나
            # 3. SSIM 평균이 임계값 이상이면 통과
            if image_eval_result.all_images_present:
                images_passed = True
            elif (
                image_eval_result.average_similarity is not None
                and image_eval_result.average_similarity >= ssim_threshold
            ):
                images_passed = True
            elif (
                image_eval_result.average_llm_score is not None
                and image_eval_result.average_llm_score >= pass_threshold
            ):
                images_passed = True

        overall_passed = text_passed and images_passed
    else:
        overall_passed = text_passed

    final_result = {
        "task_id": task_id,
        "attempt_num": attempt_num,
        "timestamp": datetime.now().isoformat(),
        "execution_time": execution_time,
        "agent_execution_time": (
            agent_execution_time if "agent_execution_time" in locals() else 0
        ),
        "summary": {
            "overall_passed": overall_passed,
            "text_evaluation_passed": text_passed,
            "image_evaluation_passed": images_passed if task.images else None,
        },
        "text_evaluation": eval_result.to_dict() if eval_result else None,
        "image_evaluation": image_eval_result.to_dict() if image_eval_result else None,
        "metadata": {
            "category": task.category,
            "difficulty": task.difficulty,
            "num_expected_images": len(task.images),
            "num_input_data": len(task.input_data),
        },
    }

    # 8. 결과 저장
    evaluation_file = task_output_dir / "evaluation.json"
    with open(evaluation_file, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(
        f"{'✅ PASSED' if overall_passed else '❌ FAILED'}: {task_id} (Attempt {attempt_num}/{total_attempts})"
    )
    print(f"{'='*80}")
    print(f"⏱️  Execution Time: {execution_time:.1f}s")
    print(f"📄 Evaluation saved to: {evaluation_file}")
    print()

    return final_result


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="HITS AI Agent QA Single Task Runner")

    parser.add_argument(
        "--task-id",
        type=str,
        required=True,
        help="Task ID to run",
    )

    parser.add_argument(
        "--attempt",
        type=int,
        required=True,
        help="Attempt number",
    )

    parser.add_argument(
        "--total-attempts",
        type=int,
        required=True,
        help="Total number of attempts",
    )

    parser.add_argument(
        "--qa-datasets-dir",
        type=str,
        required=True,
        help="QA datasets directory",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory",
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

    args = parser.parse_args()

    # 경로 변환
    qa_datasets_dir = Path(args.qa_datasets_dir)
    output_dir = Path(args.output_dir)

    # 실행
    result = run_single_task(
        task_id=args.task_id,
        attempt_num=args.attempt,
        total_attempts=args.total_attempts,
        qa_datasets_dir=qa_datasets_dir,
        output_dir=output_dir,
        pass_threshold=args.pass_threshold,
        ssim_threshold=args.ssim_threshold,
    )

    # 결과에 따라 exit code 설정
    exit_code = 0 if result.get("summary", {}).get("overall_passed", False) else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
