"""
QA Manager: QA 태스크 관리 및 로딩
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QATask:
    """QA 태스크 데이터 클래스"""

    task_id: str
    question: str
    answer: str
    category: str = "general"
    difficulty: str = "medium"
    images: List[str] = field(default_factory=list)
    input_data: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    task_path: Optional[Path] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)


class QAManager:
    """QA 태스크 관리 클래스"""

    def __init__(self, qa_datasets_dir: str):
        """
        QA Manager 초기화

        Args:
            qa_datasets_dir: QA 데이터셋 디렉토리 경로
        """
        self.qa_datasets_dir = Path(qa_datasets_dir)
        self.tasks: Dict[str, QATask] = {}
        self.load_all_tasks()

    def load_all_tasks(self) -> None:
        """모든 QA 태스크를 로딩"""
        if not self.qa_datasets_dir.exists():
            print(f"Warning: QA datasets directory does not exist: {self.qa_datasets_dir}")
            return

        task_dirs = [d for d in self.qa_datasets_dir.iterdir() if d.is_dir()]

        for task_dir in sorted(task_dirs):
            try:
                task = self._load_task_from_dir(task_dir)
                self.tasks[task.task_id] = task
            except Exception as e:
                print(f"Error loading task from {task_dir}: {e}")

        print(f"Loaded {len(self.tasks)} QA tasks")

    def _load_task_from_dir(self, task_dir: Path) -> QATask:
        """
        디렉토리에서 QA 태스크 로딩

        Args:
            task_dir: 태스크 디렉토리 경로

        Returns:
            QATask 객체
        """
        task_id = task_dir.name

        # question.md 로딩
        question_file = task_dir / "question.md"
        if not question_file.exists():
            raise FileNotFoundError(f"question.md not found in {task_dir}")
        question = question_file.read_text(encoding="utf-8")

        # answer.md 로딩
        answer_file = task_dir / "answer.md"
        if not answer_file.exists():
            raise FileNotFoundError(f"answer.md not found in {task_dir}")
        answer = answer_file.read_text(encoding="utf-8")

        # 이미지 파일 찾기 (task 폴더 바로 아래)
        images = []
        image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".svg"]
        
        # input data 파일 찾기 (task 폴더 바로 아래)
        input_data = []
        input_data_extensions = [".csv", ".txt", ".tsv", ".xlsx", ".fasta", ".fastq", ".bam", ".vcf", ".bed", ".json", ".xml"]
        excluded_files = ["question.md", "answer.md", "metadata.json"]
        
        for item in sorted(task_dir.iterdir()):
            if item.is_file():
                if item.suffix.lower() in image_extensions:
                    images.append(item.name)
                elif item.suffix.lower() in input_data_extensions and item.name not in excluded_files:
                    input_data.append(item.name)

        # 메타데이터 로딩 (있는 경우)
        metadata_file = task_dir / "metadata.json"
        metadata = {}
        category = "general"
        difficulty = "medium"
        created_at = None

        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                category = metadata.get("category", "general")
                difficulty = metadata.get("difficulty", "medium")
                created_at_str = metadata.get("created_at")
                if created_at_str:
                    created_at = datetime.fromisoformat(created_at_str)

        return QATask(
            task_id=task_id,
            question=question,
            answer=answer,
            category=category,
            difficulty=difficulty,
            images=images,
            input_data=input_data,
            metadata=metadata,
            created_at=created_at,
            task_path=task_dir,
        )

    def get_task(self, task_id: str) -> Optional[QATask]:
        """
        특정 태스크 가져오기

        Args:
            task_id: 태스크 ID

        Returns:
            QATask 객체 또는 None
        """
        return self.tasks.get(task_id)

    def list_tasks(
        self, category: Optional[str] = None, difficulty: Optional[str] = None
    ) -> List[QATask]:
        """
        태스크 목록 가져오기

        Args:
            category: 카테고리 필터 (optional)
            difficulty: 난이도 필터 (optional)

        Returns:
            QATask 리스트
        """
        tasks = list(self.tasks.values())

        if category:
            tasks = [t for t in tasks if t.category == category]
        if difficulty:
            tasks = [t for t in tasks if t.difficulty == difficulty]

        return sorted(tasks, key=lambda t: t.task_id)

    def get_task_count(self) -> int:
        """전체 태스크 개수 반환"""
        return len(self.tasks)

    def get_categories(self) -> List[str]:
        """모든 카테고리 목록 반환"""
        return sorted(set(task.category for task in self.tasks.values()))

    def get_task_statistics(self) -> Dict[str, Any]:
        """태스크 통계 정보 반환"""
        categories = {}
        difficulties = {}

        for task in self.tasks.values():
            categories[task.category] = categories.get(task.category, 0) + 1
            difficulties[task.difficulty] = difficulties.get(task.difficulty, 0) + 1

        return {
            "total_tasks": len(self.tasks),
            "categories": categories,
            "difficulties": difficulties,
        }

    def __repr__(self) -> str:
        return f"QAManager(tasks={len(self.tasks)})"

