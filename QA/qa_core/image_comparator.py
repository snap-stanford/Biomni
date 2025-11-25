"""
Image Comparator: 이미지 존재 확인 및 시각적 비교 (SSIM)
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        }


class ImageComparator:
    """이미지 비교 클래스"""

    def __init__(self, ssim_threshold: float = 0.8):
        """
        ImageComparator 초기화

        Args:
            ssim_threshold: SSIM 유사도 임계값 (0.0-1.0)
        """
        self.ssim_threshold = ssim_threshold
        self.ssim_enabled = SSIM_AVAILABLE

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
        compare_visually: bool = True,
    ) -> ImageEvaluationResult:
        """
        이미지 평가 수행

        Args:
            ground_truth_markdown: 정답 마크다운
            generated_markdown: 생성된 마크다운
            ground_truth_task_dir: 정답 태스크 디렉토리
            generated_task_dir: 생성된 태스크 디렉토리
            compare_visually: SSIM을 사용한 시각적 비교 수행 여부

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
        all_images_present = len(missing_images) == 0 and len(expected_images) > 0

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

        return ImageEvaluationResult(
            expected_images=expected_images,
            found_images=found_images,
            missing_images=missing_images,
            extra_images=extra_images,
            image_comparisons=image_comparisons,
            all_images_present=all_images_present,
            average_similarity=average_similarity,
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

