"""
Image Comparator: 이미지 존재 확인 및 시각적 비교 (SSIM, LLM)
"""

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from langchain_core.messages import HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: langchain_core not available. LLM image comparison may not work.")

try:
    from PIL import Image
    import numpy as np
    from skimage.metrics import structural_similarity as ssim

    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("Warning: PIL or scikit-image not available. SSIM comparison will be disabled.")


@dataclass
class ImageEvaluationResult:
    """이미지 평가 결과 데이터 클래스"""

    expected_images: List[str]
    found_images: List[str]
    missing_images: List[str]
    extra_images: List[str]
    image_comparisons: Dict[str, Dict] = field(default_factory=dict)
    all_images_present: bool = False
    average_similarity: Optional[float] = None
    llm_image_evaluations: Dict[str, Dict] = field(default_factory=dict)
    average_llm_score: Optional[float] = None

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "expected_images": self.expected_images,
            "found_images": self.found_images,
            "missing_images": self.missing_images,
            "extra_images": self.extra_images,
            "image_comparisons": self.image_comparisons,
            "all_images_present": self.all_images_present,
            "average_similarity": self.average_similarity,
            "llm_image_evaluations": self.llm_image_evaluations,
            "average_llm_score": self.average_llm_score,
        }


class ImageComparator:
    """이미지 비교 클래스"""

    def __init__(self, ssim_threshold: float = 0.8, llm_client=None, llm_threshold: float = 70.0):
        """
        ImageComparator 초기화

        Args:
            ssim_threshold: SSIM 유사도 임계값 (0.0-1.0)
            llm_client: LLM 클라이언트 (vision-capable, optional)
            llm_threshold: LLM 평가 통과 기준 점수 (0-100)
        """
        self.ssim_threshold = ssim_threshold
        self.ssim_enabled = SSIM_AVAILABLE
        self.llm_client = llm_client
        self.llm_threshold = llm_threshold

    def extract_image_references(self, markdown_text: str) -> List[str]:
        """
        마크다운 텍스트에서 이미지 참조 추출

        Args:
            markdown_text: 마크다운 텍스트

        Returns:
            이미지 파일명 리스트
        """
        # 마크다운 이미지 패턴: ![alt](path) 또는 ![alt](path "title")
        pattern = r"!\[.*?\]\((.*?)\)"
        matches = re.findall(pattern, markdown_text)

        # 경로에서 파일명만 추출
        image_files = []
        for match in matches:
            # 공백이나 따옴표로 구분된 경우 첫 번째 부분만 사용
            path = match.split()[0].strip('"\'')
            # 경로에서 파일명 추출
            filename = Path(path).name
            if filename:
                image_files.append(filename)

        return image_files

    def verify_images_exist(
        self, image_refs: List[str], task_dir: Path
    ) -> Tuple[List[str], List[str]]:
        """
        이미지 파일 존재 확인

        Args:
            image_refs: 참조된 이미지 파일명 리스트
            task_dir: 태스크 디렉토리 경로 (이미지가 바로 아래 있음)

        Returns:
            (found_images, missing_images) 튜플
        """
        found = []
        missing = []

        for img_ref in image_refs:
            img_path = task_dir / img_ref
            if img_path.exists():
                found.append(img_ref)
            else:
                missing.append(img_ref)

        return found, missing

    def evaluate_images(
        self,
        ground_truth_markdown: str,
        generated_markdown: str,
        ground_truth_task_dir: Path,
        generated_task_dir: Path,
        question: Optional[str] = None,
        compare_visually: bool = True,
        use_llm_comparison: bool = True,
    ) -> ImageEvaluationResult:
        """
        이미지 평가 수행

        Args:
            ground_truth_markdown: 정답 마크다운
            generated_markdown: 생성된 마크다운
            ground_truth_task_dir: 정답 태스크 디렉토리
            generated_task_dir: 생성된 태스크 디렉토리
            question: 원래 질문 (LLM 평가 시 맥락 제공)
            compare_visually: SSIM을 사용한 시각적 비교 수행 여부
            use_llm_comparison: LLM을 사용한 의미론적 비교 수행 여부

        Returns:
            ImageEvaluationResult 객체
        """
        # 이미지 참조 추출
        expected_images = self.extract_image_references(ground_truth_markdown)
        referenced_images = self.extract_image_references(generated_markdown)

        # 생성된 태스크 디렉토리에서 실제 존재하는 이미지 찾기
        found_images = []
        if generated_task_dir.exists():
            found_images = [
                img.name
                for img in generated_task_dir.iterdir()
                if img.is_file() and img.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".svg"]
            ]

        # 마크다운에 참조된 이미지 중 실제로 존재하는지 확인
        referenced_and_found = [img for img in referenced_images if img in found_images]

        # 누락된 이미지 (정답에는 있지만 생성되지 않음)
        missing_images = [img for img in expected_images if img not in found_images]

        # 추가된 이미지 (정답에는 없지만 생성됨)
        extra_images = [img for img in found_images if img not in expected_images]

        # 모든 이미지가 존재하는지 확인
        # expected_images가 없으면 (이미지가 필요없는 태스크) True
        # expected_images가 있으면 missing이 없어야 True
        all_images_present = len(missing_images) == 0

        # 시각적 비교 (SSIM)
        image_comparisons = {}
        average_similarity = None

        if compare_visually and self.ssim_enabled and expected_images:
            similarities = []
            for img_name in expected_images:
                if img_name in found_images:
                    gt_path = ground_truth_task_dir / img_name
                    gen_path = generated_task_dir / img_name

                    if gt_path.exists() and gen_path.exists():
                        try:
                            similarity = self.compare_images_ssim(gt_path, gen_path)
                            image_comparisons[img_name] = {
                                "similarity": similarity,
                                "passed": similarity >= self.ssim_threshold,
                            }
                            similarities.append(similarity)
                        except Exception as e:
                            image_comparisons[img_name] = {"error": str(e)}
                    else:
                        image_comparisons[img_name] = {"error": "File not found"}
                else:
                    image_comparisons[img_name] = {"error": "Image not generated"}

            if similarities:
                average_similarity = sum(similarities) / len(similarities)

        # LLM 기반 의미론적 비교 (파일명 무관, 내용 기반)
        llm_image_evaluations = {}
        average_llm_score = None

        if use_llm_comparison and self.llm_client and expected_images and found_images:
            # 기대 이미지 경로들
            expected_image_paths = [
                ground_truth_task_dir / img_name
                for img_name in expected_images
                if (ground_truth_task_dir / img_name).exists()
            ]
            
            # 생성된 이미지 경로들
            generated_image_paths = [
                generated_task_dir / img_name
                for img_name in found_images
            ]
            
            if expected_image_paths and generated_image_paths:
                try:
                    # 모든 이미지를 한 번에 LLM에게 보여주고 평가
                    llm_result = self.compare_all_images_with_llm(
                        expected_image_paths=expected_image_paths,
                        generated_image_paths=generated_image_paths,
                        question=question,
                    )
                    llm_image_evaluations["overall"] = llm_result
                    if "score" in llm_result:
                        average_llm_score = llm_result["score"]
                        
                    # all_images_present를 LLM 평가 결과로 업데이트
                    if llm_result.get("all_content_present", False):
                        all_images_present = True
                        missing_images = []  # 내용은 모두 포함되어 있음
                        
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    llm_image_evaluations["overall"] = {"error": str(e)}

        return ImageEvaluationResult(
            expected_images=expected_images,
            found_images=found_images,
            missing_images=missing_images,
            extra_images=extra_images,
            image_comparisons=image_comparisons,
            all_images_present=all_images_present,
            average_similarity=average_similarity,
            llm_image_evaluations=llm_image_evaluations,
            average_llm_score=average_llm_score,
        )

    def compare_images_ssim(self, image1_path: Path, image2_path: Path) -> float:
        """
        SSIM을 사용한 이미지 유사도 비교

        Args:
            image1_path: 첫 번째 이미지 경로
            image2_path: 두 번째 이미지 경로

        Returns:
            SSIM 유사도 점수 (0.0-1.0)
        """
        if not self.ssim_enabled:
            raise RuntimeError("SSIM comparison is not available. Install PIL and scikit-image.")

        # 이미지 로드
        img1 = Image.open(image1_path).convert("RGB")
        img2 = Image.open(image2_path).convert("RGB")

        # 크기가 다르면 리사이즈
        if img1.size != img2.size:
            # 더 작은 크기로 맞춤
            target_size = (
                min(img1.size[0], img2.size[0]),
                min(img1.size[1], img2.size[1]),
            )
            img1 = img1.resize(target_size, Image.LANCZOS)
            img2 = img2.resize(target_size, Image.LANCZOS)

        # numpy 배열로 변환
        arr1 = np.array(img1)
        arr2 = np.array(img2)

        # SSIM 계산 (multichannel for RGB)
        similarity = ssim(arr1, arr2, channel_axis=2, data_range=255)

        return float(similarity)

    def compare_all_images_with_llm(
        self,
        expected_image_paths: List[Path],
        generated_image_paths: List[Path],
        question: Optional[str] = None,
    ) -> Dict:
        """
        LLM을 사용한 모든 이미지 내용 비교 (파일명 무관)
        
        여러 기대 이미지의 내용이 생성된 이미지들에 모두 포함되어 있는지 확인.
        생성된 이미지가 여러 그래프를 하나로 합쳤을 수도 있음.

        Args:
            expected_image_paths: 기대 이미지 경로 리스트
            generated_image_paths: 생성된 이미지 경로 리스트
            question: 원래 질문 (맥락 제공)

        Returns:
            평가 결과 딕셔너리 (score, feedback, passed, all_content_present)
        """
        if not self.llm_client:
            return {"error": "LLM client not available"}

        try:
            # 평가 프롬프트 생성
            prompt = self._create_multi_image_comparison_prompt(
                question=question,
                num_expected=len(expected_image_paths),
                num_generated=len(generated_image_paths),
            )

            # 컨텐츠 구성: 프롬프트 + 기대 이미지들 + 생성 이미지들
            content_parts = [{"type": "text", "text": prompt}]
            
            # 기대 이미지들 추가
            content_parts.append({
                "type": "text",
                "text": f"\n**기대되는 이미지들 (Expected Images, {len(expected_image_paths)}개):**\n"
            })
            for idx, img_path in enumerate(expected_image_paths, 1):
                img_base64 = self._encode_image_to_base64(img_path)
                content_parts.append({
                    "type": "text",
                    "text": f"\n기대 이미지 {idx}: `{img_path.name}`\n"
                })
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                })
            
            # 생성된 이미지들 추가
            content_parts.append({
                "type": "text",
                "text": f"\n\n**생성된 이미지들 (Generated Images, {len(generated_image_paths)}개):**\n"
            })
            for idx, img_path in enumerate(generated_image_paths, 1):
                img_base64 = self._encode_image_to_base64(img_path)
                content_parts.append({
                    "type": "text",
                    "text": f"\n생성 이미지 {idx}: `{img_path.name}`\n"
                })
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                })

            # LLM 호출 (LangChain 메시지 포맷)
            if not LANGCHAIN_AVAILABLE:
                return {"error": "langchain_core not available"}
            
            message = HumanMessage(content=content_parts)
            
            if hasattr(self.llm_client, "invoke"):
                response = self.llm_client.invoke([message])
            else:
                response = self.llm_client([message])

            # 응답 파싱
            if isinstance(response, str):
                response_text = response
            elif hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            # JSON 파싱
            result = self._parse_llm_multi_image_response(response_text)
            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"LLM multi-image comparison failed: {str(e)}"}

    def compare_images_with_llm(
        self,
        ground_truth_path: Path,
        generated_path: Path,
        question: Optional[str] = None,
        image_name: Optional[str] = None,
    ) -> Dict:
        """
        LLM을 사용한 이미지 의미론적 비교

        Args:
            ground_truth_path: 정답 이미지 경로
            generated_path: 생성된 이미지 경로
            question: 원래 질문 (맥락 제공)
            image_name: 이미지 파일명

        Returns:
            평가 결과 딕셔너리 (score, feedback, passed)
        """
        if not self.llm_client:
            return {"error": "LLM client not available"}

        try:
            # 이미지를 base64로 인코딩
            gt_base64 = self._encode_image_to_base64(ground_truth_path)
            gen_base64 = self._encode_image_to_base64(generated_path)

            # 평가 프롬프트 생성
            prompt = self._create_image_comparison_prompt(question, image_name)

            # LLM 호출 (vision-capable)
            content_parts = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{gt_base64}"},
                },
                {"type": "text", "text": "\n**생성된 이미지 (Generated Image):**\n"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{gen_base64}"},
                },
            ]

            # LLM 호출 (LangChain 메시지 포맷)
            if not LANGCHAIN_AVAILABLE:
                return {"error": "langchain_core not available"}
            
            message = HumanMessage(content=content_parts)
            
            if hasattr(self.llm_client, "invoke"):
                response = self.llm_client.invoke([message])
            else:
                response = self.llm_client([message])

            # 응답 파싱
            if isinstance(response, str):
                response_text = response
            elif hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            # JSON 파싱
            result = self._parse_llm_image_response(response_text)
            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"LLM image comparison failed: {str(e)}"}

    def _encode_image_to_base64(self, image_path: Path) -> str:
        """
        이미지를 base64로 인코딩

        Args:
            image_path: 이미지 파일 경로

        Returns:
            base64 인코딩된 문자열
        """
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _create_multi_image_comparison_prompt(
        self,
        question: Optional[str] = None,
        num_expected: int = 0,
        num_generated: int = 0,
    ) -> str:
        """
        다중 이미지 비교를 위한 LLM 프롬프트 생성

        Args:
            question: 원래 질문
            num_expected: 기대 이미지 개수
            num_generated: 생성 이미지 개수

        Returns:
            프롬프트 문자열
        """
        prompt_parts = [
            "You are an expert evaluator for scientific visualization and data analysis images.",
            "",
            "**Task:** Evaluate if the generated images contain all the information/content from the expected images.",
            "",
            "**IMPORTANT NOTES:**",
            "- The generated image filenames may be DIFFERENT from expected filenames",
            "- Multiple expected graphs MAY BE COMBINED into a single generated image (e.g., boxplot + histogram → combined_plot.png)",
            "- A single expected graph may also be split across multiple generated images",
            "- Focus on CONTENT, not filenames or exact visual styling",
            "",
        ]

        if question:
            prompt_parts.extend(
                [
                    "**Original Question/Task:**",
                    f"{question}",
                    "",
                ]
            )

        prompt_parts.extend(
            [
                f"**Expected Images:** {num_expected} image(s)",
                f"**Generated Images:** {num_generated} image(s)",
                "",
                "**Evaluation Criteria:**",
                "",
                "1. **Content Completeness (50%)**: Do the generated images contain ALL the information from ALL expected images?",
                "   - Check if all data, variables, and key information are present",
                "   - It's OK if multiple expected images are combined into one",
                "",
                "2. **Visualization Type (30%)**: Are appropriate chart types used?",
                "   - Verify that the visualization types match the requirements",
                "   - Bar plots, line plots, scatter plots, boxplots, histograms, etc.",
                "",
                "3. **Key Features (20%)**: Are critical visual elements preserved?",
                "   - Trends, patterns, distributions",
                "   - Data ranges and scales",
                "",
                "**Note:** Differences in colors, fonts, styling, or file names are acceptable.",
                "",
                "Respond in the following JSON format ONLY:",
                "```json",
                "{",
                '  "score": <0-100>,',
                '  "all_content_present": <true/false>,',
                '  "feedback": "<detailed explanation in Korean>",',
                '  "matching_details": "<which generated images correspond to which expected images>"',
                "}",
                "```",
                "",
            ]
        )

        return "\n".join(prompt_parts)

    def _create_image_comparison_prompt(
        self, question: Optional[str] = None, image_name: Optional[str] = None
    ) -> str:
        """
        이미지 비교를 위한 LLM 프롬프트 생성

        Args:
            question: 원래 질문
            image_name: 이미지 파일명

        Returns:
            프롬프트 문자열
        """
        prompt_parts = [
            "You are an expert evaluator for scientific visualization and data analysis images.",
            "",
            "**Task:** Compare the two images below and evaluate if they convey the same information and meaning.",
            "",
        ]

        if question:
            prompt_parts.extend(
                [
                    "**Original Question/Task:**",
                    f"{question}",
                    "",
                ]
            )

        if image_name:
            prompt_parts.extend(
                [
                    f"**Image Name:** {image_name}",
                    "",
                ]
            )

        prompt_parts.extend(
            [
                "**Evaluation Criteria:**",
                "",
                "1. **Data Content (40%)**: Do both images show the same data/information?",
                "   - Check if the same variables, categories, or data points are present",
                "   - Verify that numerical values and ranges are consistent",
                "",
                "2. **Visual Representation (30%)**: Is the type of visualization appropriate and similar?",
                "   - Chart type (bar plot, line plot, scatter plot, boxplot, etc.)",
                "   - Overall structure and layout",
                "",
                "3. **Key Features (30%)**: Are the critical visual elements preserved?",
                "   - Trends, patterns, distributions",
                "   - Labels, legends, titles (if critical to understanding)",
                "   - Color schemes and visual encoding (less critical if data is same)",
                "",
                "**Note:** Minor differences in styling, colors, fonts, or exact positioning are acceptable",
                "if the core data and meaning are preserved.",
                "",
                "**First Image is the Ground Truth (Expected), Second Image is Generated (To Evaluate)**",
                "",
                "Respond in the following JSON format ONLY:",
                "```json",
                "{",
                '  "score": <0-100>,',
                '  "feedback": "<brief explanation in Korean about similarities and differences>"',
                "}",
                "```",
                "",
                "**정답 이미지 (Ground Truth):**",
            ]
        )

        prompt = "\n".join(prompt_parts)
        return prompt

    def _parse_llm_multi_image_response(self, response: str) -> Dict:
        """
        LLM 다중 이미지 비교 응답 파싱

        Args:
            response: LLM 응답 텍스트

        Returns:
            파싱된 결과 딕셔너리
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
            score = float(result.get("score", 0))
            feedback = result.get("feedback", "No feedback provided")
            all_content_present = result.get("all_content_present", False)
            matching_details = result.get("matching_details", "")
            passed = score >= self.llm_threshold

            return {
                "score": score,
                "feedback": feedback,
                "all_content_present": all_content_present,
                "matching_details": matching_details,
                "passed": passed,
            }
        except json.JSONDecodeError:
            # 폴백: 텍스트에서 점수 추출 시도
            score_match = re.search(r"score[:\s]+(\d+(?:\.\d+)?)", response, re.IGNORECASE)
            content_match = re.search(
                r"all_content_present[:\s]+(true|false)", response, re.IGNORECASE
            )
            
            score = float(score_match.group(1)) if score_match else 0.0
            all_content_present = (
                content_match.group(1).lower() == "true" if content_match else False
            )
            passed = score >= self.llm_threshold
            
            return {
                "score": score,
                "feedback": response[:500],
                "all_content_present": all_content_present,
                "matching_details": "",
                "passed": passed,
            }

    def _parse_llm_image_response(self, response: str) -> Dict:
        """
        LLM 이미지 비교 응답 파싱

        Args:
            response: LLM 응답 텍스트

        Returns:
            파싱된 결과 딕셔너리
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
            score = float(result.get("score", 0))
            feedback = result.get("feedback", "No feedback provided")
            passed = score >= self.llm_threshold

            return {
                "score": score,
                "feedback": feedback,
                "passed": passed,
            }
        except json.JSONDecodeError:
            # 폴백: 텍스트에서 점수 추출 시도
            score_match = re.search(r"score[:\s]+(\d+(?:\.\d+)?)", response, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                passed = score >= self.llm_threshold
                return {
                    "score": score,
                    "feedback": response[:500],
                    "passed": passed,
                }
            else:
                return {
                    "score": 0.0,
                    "feedback": f"Failed to parse response: {response[:200]}",
                    "passed": False,
                }

    def get_image_info(self, image_path: Path) -> Dict:
        """
        이미지 메타정보 추출

        Args:
            image_path: 이미지 파일 경로

        Returns:
            이미지 정보 딕셔너리
        """
        if not SSIM_AVAILABLE:
            return {"error": "PIL not available"}

        try:
            img = Image.open(image_path)
            return {
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.width,
                "height": img.height,
            }
        except Exception as e:
            return {"error": str(e)}

