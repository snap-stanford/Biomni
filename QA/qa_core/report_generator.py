"""
Report Generator: 평가 결과 리포트 생성
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .evaluator import EvaluationResult
from .image_comparator import ImageEvaluationResult


class ReportGenerator:
    """평가 결과 리포트 생성 클래스"""

    def __init__(self):
        """ReportGenerator 초기화"""
        pass

    def generate_task_report(
        self,
        task_id: str,
        evaluation_result: EvaluationResult,
        image_evaluation: ImageEvaluationResult,
        execution_time: float,
        output_path: Path,
    ) -> None:
        """
        개별 태스크 평가 리포트 생성 (JSON)

        Args:
            task_id: 태스크 ID
            evaluation_result: 텍스트 평가 결과
            image_evaluation: 이미지 평가 결과
            execution_time: 실행 시간 (초)
            output_path: 출력 파일 경로
        """
        # LLM 이미지 평가 결과 확인
        llm_image_passed = True
        llm_evals = image_evaluation.llm_image_evaluations
        expected_images = image_evaluation.expected_images
        
        # 이미지가 필요한 태스크인 경우에만 LLM 평가 체크
        if expected_images:
            if not llm_evals:
                # 이미지가 필요한데 LLM 평가가 없으면 실패
                llm_image_passed = False
            elif "overall" in llm_evals:
                # 다중 이미지 비교 결과
                llm_result = llm_evals["overall"]
                if "error" in llm_result:
                    # 에러가 있으면 실패
                    llm_image_passed = False
                elif not llm_result.get("passed", False):
                    # passed가 False면 실패
                    llm_image_passed = False
            else:
                # 개별 이미지 비교: 하나라도 실패하면 전체 실패
                for img_name, llm_result in llm_evals.items():
                    if "error" in llm_result or not llm_result.get("passed", False):
                        llm_image_passed = False
                        break
        
        report = {
            "task_id": task_id,
            "timestamp": evaluation_result.timestamp.isoformat(),
            "execution_time_seconds": execution_time,
            "text_evaluation": {
                "scores": evaluation_result.scores,
                "overall_score": evaluation_result.overall_score,
                "passed": evaluation_result.passed,
                "llm_feedback": evaluation_result.llm_feedback,
                "metadata": evaluation_result.metadata,
            },
            "image_evaluation": image_evaluation.to_dict(),
            "summary": {
                "overall_passed": (
                    evaluation_result.passed 
                    and image_evaluation.all_images_present 
                    and llm_image_passed
                ),
                "text_score": evaluation_result.overall_score,
                "images_present": image_evaluation.all_images_present,
                "average_image_similarity": image_evaluation.average_similarity,
                "llm_image_passed": llm_image_passed,
            },
        }

        # JSON 파일로 저장
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Task report saved to: {output_path}")

    def generate_summary_report(
        self, all_results: List[Dict[str, Any]], output_path: Path
    ) -> None:
        """
        전체 태스크 종합 리포트 생성 (Markdown)

        Args:
            all_results: 모든 태스크의 평가 결과 리스트
            output_path: 출력 파일 경로
        """
        # 통계 계산
        total_tasks = len(all_results)
        passed_tasks = sum(1 for r in all_results if r.get("summary", {}).get("overall_passed", False))
        avg_score = (
            sum(r.get("text_evaluation", {}).get("overall_score", 0) for r in all_results) / total_tasks
            if total_tasks > 0
            else 0
        )

        # 마크다운 리포트 생성
        report_lines = [
            "# HITS AI Agent QA 평가 종합 리포트",
            "",
            f"**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**총 태스크 수**: {total_tasks}",
            f"**통과 태스크**: {passed_tasks}/{total_tasks} ({passed_tasks/total_tasks*100:.1f}%)",
            f"**평균 점수**: {avg_score:.2f}/100",
            "",
            "---",
            "",
            "## 태스크별 결과",
            "",
            "| Task ID | 텍스트 점수 | 이미지(LLM) | Content✓ | 통과 여부 | 피드백 |",
            "|---------|-------------|-------------|----------|-----------|--------|",
        ]

        for result in sorted(all_results, key=lambda x: x["task_id"]):
            task_id = result["task_id"]
            text_eval = result.get("text_evaluation", {})
            img_eval = result.get("image_evaluation", {})
            summary = result.get("summary", {})

            text_score = text_eval.get("overall_score", 0)
            content_present = img_eval.get("all_images_present", False)
            avg_llm = img_eval.get("average_llm_score")
            passed = summary.get("overall_passed", False)
            feedback = text_eval.get("llm_feedback", "")[:50]  # 처음 50자만

            status_icon = "✅" if passed else "❌"
            content_icon = "✅" if content_present else "❌"
            
            # LLM 이미지 점수 표시
            llm_display = f"{avg_llm:.0f}" if avg_llm is not None else "N/A"

            report_lines.append(
                f"| {task_id} | {text_score:.1f} | {llm_display} | {content_icon} | {status_icon} | {feedback}... |"
            )

        report_lines.extend(
            [
                "",
                "---",
                "",
                "## 세부 통계",
                "",
                "### 점수 분포",
                "",
            ]
        )

        # 점수 분포 계산
        score_ranges = {"90-100": 0, "80-89": 0, "70-79": 0, "60-69": 0, "0-59": 0}
        for result in all_results:
            score = result.get("text_evaluation", {}).get("overall_score", 0)
            if score >= 90:
                score_ranges["90-100"] += 1
            elif score >= 80:
                score_ranges["80-89"] += 1
            elif score >= 70:
                score_ranges["70-79"] += 1
            elif score >= 60:
                score_ranges["60-69"] += 1
            else:
                score_ranges["0-59"] += 1

        for range_name, count in score_ranges.items():
            report_lines.append(f"- **{range_name}점**: {count}개 태스크")

        report_lines.extend(
            [
                "",
                "### 이미지 평가 (LLM Content-Based)",
                "",
            ]
        )

        # 이미지 통계 (LLM 기반)
        tasks_with_images = sum(
            1 for r in all_results if len(r.get("image_evaluation", {}).get("expected_images", [])) > 0
        )
        tasks_all_content_present = sum(
            1 for r in all_results if r.get("image_evaluation", {}).get("all_images_present", False)
        )

        report_lines.append(f"- **이미지 필요 태스크**: {tasks_with_images}개")
        report_lines.append(
            f"- **모든 내용 포함 (LLM 판단)**: {tasks_all_content_present}/{tasks_with_images}개"
        )

        # LLM 이미지 평가 평균
        llm_scores = [
            r.get("image_evaluation", {}).get("average_llm_score")
            for r in all_results
            if r.get("image_evaluation", {}).get("average_llm_score") is not None
        ]
        if llm_scores:
            avg_llm_score = sum(llm_scores) / len(llm_scores)
            report_lines.append(f"- **평균 LLM 이미지 점수**: {avg_llm_score:.1f}/100")
        
        # SSIM은 참고용으로만
        ssim_scores = [
            r.get("image_evaluation", {}).get("average_similarity")
            for r in all_results
            if r.get("image_evaluation", {}).get("average_similarity") is not None
        ]
        if ssim_scores:
            avg_ssim = sum(ssim_scores) / len(ssim_scores)
            report_lines.append(f"- **참고: SSIM (파일명 매칭시)**: {avg_ssim:.3f}")

        report_lines.extend(["", "---", "", f"*리포트 생성: HITS AI Agent QA System*", ""])

        # 파일로 저장
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print(f"Summary report saved to: {output_path}")

    def load_task_report(self, report_path: Path) -> Dict[str, Any]:
        """
        태스크 리포트 로드

        Args:
            report_path: 리포트 파일 경로

        Returns:
            리포트 딕셔너리
        """
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def print_task_summary(self, report: Dict[str, Any]) -> None:
        """
        태스크 리포트 요약 출력

        Args:
            report: 리포트 딕셔너리
        """
        print(f"\n{'='*60}")
        print(f"Task: {report['task_id']}")
        print(f"{'='*60}")

        text_eval = report.get("text_evaluation", {})
        print(f"\n📊 Text Evaluation:")
        print(f"  - Overall Score: {text_eval.get('overall_score', 0):.1f}/100")
        print(f"  - Content Accuracy: {text_eval.get('scores', {}).get('content_accuracy', 0):.1f}")
        print(f"  - Completeness: {text_eval.get('scores', {}).get('completeness', 0):.1f}")
        print(f"  - Format Compliance: {text_eval.get('scores', {}).get('format_compliance', 0):.1f}")
        print(f"  - Passed: {'✅ Yes' if text_eval.get('passed', False) else '❌ No'}")

        img_eval = report.get("image_evaluation", {})
        print(f"\n🖼️  Image Evaluation:")
        
        # LLM 기반 평가 (주 평가)
        if img_eval.get("average_llm_score") is not None:
            print(f"\n  [LLM Content-Based Evaluation - Primary]")
            print(f"  - LLM Score: {img_eval.get('average_llm_score', 0):.1f}/100")
            print(
                f"  - All Content Present: {'✅ Yes' if img_eval.get('all_images_present', False) else '❌ No'}"
            )
        else:
            print(f"\n  [Basic Check]")
            print(f"  - Expected Images (by name): {len(img_eval.get('expected_images', []))}")
            print(f"  - Found Images: {len(img_eval.get('found_images', []))}")
            print(
                f"  - All Images Present: {'✅ Yes' if img_eval.get('all_images_present', False) else '❌ No'}"
            )
        
        # 참고 정보
        print(f"\n  [Reference Info]")
        print(f"  - Expected filenames: {img_eval.get('expected_images', [])}")
        print(f"  - Generated filenames: {img_eval.get('found_images', [])}")
        if img_eval.get("average_similarity") is not None:
            print(f"  - SSIM (if name matched): {img_eval.get('average_similarity', 0):.3f}")
            
        # LLM 이미지 평가 세부 내용
        llm_evals = img_eval.get("llm_image_evaluations", {})
        if llm_evals:
            print(f"\n  📊 LLM Image Content Comparison:")
            # "overall" 키가 있으면 다중 이미지 비교 결과
            if "overall" in llm_evals:
                llm_result = llm_evals["overall"]
                if "error" in llm_result:
                    print(f"    ❌ Error: {llm_result['error']}")
                else:
                    score = llm_result.get("score", 0)
                    passed = llm_result.get("passed", False)
                    all_content = llm_result.get("all_content_present", False)
                    status = "✅" if passed else "❌"
                    content_status = "✅" if all_content else "❌"
                    
                    print(f"    • Overall Score: {status} {score:.1f}/100")
                    print(f"    • All Content Present: {content_status}")
                    
                    matching = llm_result.get("matching_details", "")
                    if matching:
                        print(f"    • Matching: {matching}")
                    
                    feedback = llm_result.get("feedback", "")
                    if feedback:
                        # 피드백을 짧게 출력 (첫 150자)
                        short_feedback = feedback[:150] + "..." if len(feedback) > 150 else feedback
                        print(f"    • Feedback: {short_feedback}")
            else:
                # 개별 이미지 비교 결과 (구버전 호환)
                for img_name, llm_result in llm_evals.items():
                    if "error" in llm_result:
                        print(f"    • {img_name}: ❌ {llm_result['error']}")
                    else:
                        score = llm_result.get("score", 0)
                        passed = llm_result.get("passed", False)
                        status = "✅" if passed else "❌"
                        print(f"    • {img_name}: {status} {score:.1f}/100")
                        feedback = llm_result.get("feedback", "")
                        if feedback:
                            # 피드백을 짧게 출력 (첫 100자)
                            short_feedback = feedback[:100] + "..." if len(feedback) > 100 else feedback
                            print(f"      → {short_feedback}")

        print(f"\n💬 LLM Feedback:")
        print(f"  {text_eval.get('llm_feedback', 'No feedback')}")

        summary = report.get("summary", {})
        overall_passed = summary.get("overall_passed", False)
        llm_image_passed = summary.get("llm_image_passed", True)
        
        print(f"\n{'='*60}")
        print(f"Overall Result: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
        
        # 실패 원인 표시
        if not overall_passed:
            reasons = []
            if not text_eval.get("passed", False):
                reasons.append("Text evaluation failed")
            if not img_eval.get("all_images_present", False):
                reasons.append("Image content incomplete (LLM)")
            if not llm_image_passed:
                reasons.append("LLM image evaluation failed/error")
            if reasons:
                print(f"Failure reasons: {', '.join(reasons)}")
        
        print(f"{'='*60}\n")

