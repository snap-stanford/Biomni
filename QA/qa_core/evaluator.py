"""
Evaluator: LLM 기반 답변 평가
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class EvaluationResult:
    """평가 결과 데이터 클래스"""

    task_id: str
    timestamp: datetime
    scores: Dict[str, float]  # content_accuracy, completeness, format_compliance
    overall_score: float
    llm_feedback: str
    passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
            "scores": self.scores,
            "overall_score": self.overall_score,
            "llm_feedback": self.llm_feedback,
            "passed": self.passed,
            "metadata": self.metadata,
        }


class Evaluator:
    """LLM 기반 답변 평가 클래스"""

    def __init__(self, llm_client, pass_threshold: float = 70.0, evaluation_prompt_path: Optional[str] = None):
        """
        Evaluator 초기화

        Args:
            llm_client: LLM 클라이언트 (biomni.llm 등)
            pass_threshold: 통과 기준 점수 (0-100)
            evaluation_prompt_path: 평가 프롬프트 파일 경로 (optional)
        """
        self.llm_client = llm_client
        self.pass_threshold = pass_threshold
        self.evaluation_prompt_template = self._load_evaluation_prompt(evaluation_prompt_path)

    def _load_evaluation_prompt(self, prompt_path: Optional[str]) -> str:
        """평가 프롬프트 로딩"""
        if prompt_path:
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                print(f"Warning: Could not load evaluation prompt from {prompt_path}: {e}")

        # 기본 프롬프트
        return """You are an expert evaluator for a biological AI agent's responses.

<GROUND_TRUTH>
{ground_truth}
</GROUND_TRUTH>

<GENERATED_ANSWER>
{generated_answer}
</GENERATED_ANSWER>

Please evaluate the generated answer against the ground truth on the following criteria:

1. **Content Accuracy (0-100)**: How accurate is the generated answer compared to the ground truth?
   - Consider factual correctness, key concepts, and scientific accuracy
   - 90-100: Nearly identical or equally correct
   - 70-89: Mostly correct with minor differences
   - 50-69: Partially correct but missing key information
   - 0-49: Incorrect or significantly different

2. **Completeness (0-100)**: Does the generated answer cover all requirements?
   - Consider whether all questions are answered
   - Check if all expected sections/components are present
   - Verify that the depth of explanation is adequate

3. **Format Compliance (0-100)**: Does the generated answer follow proper markdown formatting?
   - Check markdown syntax correctness
   - Verify proper use of headers, lists, code blocks
   - Assess overall readability and structure

4. **Overall Assessment**: Provide brief feedback on strengths and weaknesses

Respond in the following JSON format ONLY:
```json
{{
  "content_accuracy": <score 0-100>,
  "completeness": <score 0-100>,
  "format_compliance": <score 0-100>,
  "feedback": "<brief feedback in Korean>"
}}
```"""

    def evaluate_answer(
        self, task_id: str, question: str, ground_truth: str, generated_answer: str
    ) -> EvaluationResult:
        """
        답변 평가 수행

        Args:
            task_id: 태스크 ID
            question: 질문 텍스트
            ground_truth: 정답 텍스트
            generated_answer: 생성된 답변 텍스트

        Returns:
            EvaluationResult 객체
        """
        # 프롬프트 생성
        evaluation_prompt = self.evaluation_prompt_template.format(
            ground_truth=ground_truth, generated_answer=generated_answer
        )

        # LLM 호출
        try:
            # LangChain의 invoke 메서드 사용 (최신 방식)
            if hasattr(self.llm_client, 'invoke'):
                llm_response = self.llm_client.invoke(evaluation_prompt)
            else:
                llm_response = self.llm_client(evaluation_prompt)
            
            # 응답 타입에 따라 처리
            if isinstance(llm_response, str):
                response_text = llm_response
            elif hasattr(llm_response, 'content'):
                response_text = llm_response.content
            else:
                response_text = str(llm_response)
            
            scores, feedback = self._parse_llm_response(response_text)
        except Exception as e:
            print(f"Error calling LLM for evaluation: {e}")
            import traceback
            traceback.print_exc()
            # 폴백: 기본 점수 반환
            scores = {"content_accuracy": 0.0, "completeness": 0.0, "format_compliance": 0.0}
            feedback = f"LLM evaluation failed: {str(e)}"

        # 전체 점수 계산 (가중 평균)
        overall_score = (
            scores["content_accuracy"] * 0.5
            + scores["completeness"] * 0.3
            + scores["format_compliance"] * 0.2
        )

        # 통과 여부 판단
        passed = overall_score >= self.pass_threshold

        return EvaluationResult(
            task_id=task_id,
            timestamp=datetime.now(),
            scores=scores,
            overall_score=overall_score,
            llm_feedback=feedback,
            passed=passed,
            metadata={"question_length": len(question), "answer_length": len(generated_answer)},
        )

    def _parse_llm_response(self, response: str) -> tuple[Dict[str, float], str]:
        """
        LLM 응답 파싱

        Args:
            response: LLM 응답 텍스트

        Returns:
            (scores, feedback) 튜플
        """
        # JSON 코드 블록 추출
        json_match = re.search(r"```json\s*\n(.*?)\n```", response, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            # JSON 코드 블록 없이 바로 JSON인 경우
            json_text = response

        try:
            result = json.loads(json_text)
            scores = {
                "content_accuracy": float(result.get("content_accuracy", 0)),
                "completeness": float(result.get("completeness", 0)),
                "format_compliance": float(result.get("format_compliance", 0)),
            }
            feedback = result.get("feedback", "No feedback provided")
            return scores, feedback
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response as JSON: {e}")
            print(f"Response: {response[:500]}...")
            # 폴백: 텍스트에서 숫자 추출 시도
            return self._fallback_parse(response)

    def _fallback_parse(self, response: str) -> tuple[Dict[str, float], str]:
        """
        폴백 파싱: 텍스트에서 숫자 패턴 추출

        Args:
            response: LLM 응답 텍스트

        Returns:
            (scores, feedback) 튜플
        """
        scores = {"content_accuracy": 0.0, "completeness": 0.0, "format_compliance": 0.0}

        # 숫자 패턴 찾기
        patterns = {
            "content_accuracy": r"content[_ ]accuracy[:\s]+(\d+)",
            "completeness": r"completeness[:\s]+(\d+)",
            "format_compliance": r"format[_ ]compliance[:\s]+(\d+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                scores[key] = float(match.group(1))

        feedback = response[:500]  # 처음 500자를 피드백으로 사용

        return scores, feedback

    def compare_markdown_structure(self, ground_truth: str, generated: str) -> Dict[str, Any]:
        """
        마크다운 구조 비교 (간단한 통계 기반)

        Args:
            ground_truth: 정답 마크다운
            generated: 생성된 마크다운

        Returns:
            구조 비교 결과 딕셔너리
        """
        gt_headers = len(re.findall(r"^#+\s", ground_truth, re.MULTILINE))
        gen_headers = len(re.findall(r"^#+\s", generated, re.MULTILINE))

        gt_code_blocks = len(re.findall(r"```", ground_truth))
        gen_code_blocks = len(re.findall(r"```", generated))

        gt_lists = len(re.findall(r"^\s*[-*+]\s", ground_truth, re.MULTILINE))
        gen_lists = len(re.findall(r"^\s*[-*+]\s", generated, re.MULTILINE))

        return {
            "headers": {"ground_truth": gt_headers, "generated": gen_headers},
            "code_blocks": {"ground_truth": gt_code_blocks // 2, "generated": gen_code_blocks // 2},
            "lists": {"ground_truth": gt_lists, "generated": gen_lists},
        }

