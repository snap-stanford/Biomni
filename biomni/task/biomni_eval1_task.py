import numpy as np
import pandas as pd

from biomni.task.base_task import base_task
from biomni.eval import BiomniEval1

np.random.seed(42)


class biomni_eval1_task(base_task):
    """
    Task loader for BiomniEval1 benchmark tasks

    Supports all tasks from BiomniEval1:
    - crispr_delivery
    - gwas_causal_gene_opentargets
    - gwas_causal_gene_pharmaprojects
    - gwas_causal_gene_gwas_catalog
    - gwas_variant_prioritization
    - lab_bench_dbqa
    - lab_bench_seqqa
    - rare_disease_diagnosis
    - screen_gene_retrieval
    - patient_gene_detection
    """

    def __init__(self, task_name: str, split: str = "val"):
        """
        Initialize the BiomniEval1 task loader

        Args:
            task_name: Name of the task (e.g., 'gwas_causal_gene_opentargets')
            split: Data split to use ('train' or 'val')
        """
        if split not in ["train", "val"]:
            raise ValueError("split must be one of 'train', 'val'")

        self.task_name = task_name
        self.split = split

        # Load BiomniEval1 dataset
        self.evaluator = BiomniEval1()

        # Get instances for this task and split
        self.df = self.evaluator.get_instances_by_task(task_name, split)

        if len(self.df) == 0:
            raise ValueError(
                f"No instances found for task {task_name} with split {split}"
            )

        # Sort by task_instance_id to ensure consistent ordering
        self.df = self.df.sort_values("task_instance_id").reset_index(drop=True)

        print(f"Loaded {len(self.df)} instances for task {task_name} (split: {split})")

        # Define answer format instructions based on task type
        self._answer_instruction = self._get_answer_instruction(task_name)

    def _get_answer_instruction(self, task_name: str) -> str:
        """
        Get task-specific answer format instruction

        Args:
            task_name: Name of the task

        Returns:
            str: Answer format instruction to append to prompts
        """
        if task_name in ["crispr_delivery", "lab_bench_db", "lab_bench_seq", "hle"]:
            # Multiple choice tasks - expect a letter
            return """

You MUST include the letter of the correct answer within the following tags:
[ANSWER] and [/ANSWER]. For example, '[ANSWER]A[/ANSWER]',
where the answer is the correct letter. Always answer in exactly this format
of a single letter between the two tags, even if you are unsure.
We require this because we use automatic parsing."""

        elif task_name.startswith("gwas_causal_gene"):
            # GWAS causal gene tasks - expect a gene symbol
            return """

You MUST include your answer (the gene symbol) within the following tags:
[ANSWER] and [/ANSWER]. For example, '[ANSWER]BRCA1[/ANSWER]',
where the answer is the gene symbol. Always answer in exactly this format
between the two tags, even if you are unsure.
We require this because we use automatic parsing."""

        elif task_name == "gwas_variant_prioritization":
            # GWAS variant prioritization - expect a variant ID
            return """

You MUST include your answer (the variant ID) within the following tags:
[ANSWER] and [/ANSWER]. For example, '[ANSWER]rs1234567[/ANSWER]',
where the answer is the variant identifier. Always answer in exactly this format
between the two tags, even if you are unsure.
We require this because we use automatic parsing.
"""

        elif task_name == "rare_disease_diagnosis":
            # Rare disease diagnosis - expect JSON with OMIM_ID
            return """

You MUST include your answer in JSON format within the following tags:
[ANSWER] and [/ANSWER]. For example, '[ANSWER]{"OMIM_ID": "123456", "Disease": "Disease Name"}[/ANSWER]',
where the answer includes the OMIM_ID and disease name. Always answer in exactly this format
between the two tags, even if you are unsure.
We require this because we use automatic parsing."""

        elif task_name == "patient_gene_detection":
            # Patient gene detection - expect JSON with causal_gene list
            return """

You MUST include your answer in JSON format within the following tags:
[ANSWER] and [/ANSWER]. For example, '[ANSWER]{"causal_gene": ["GENE1", "GENE2"]}[/ANSWER]',
where the answer includes a list of causal genes. Always answer in exactly this format
between the two tags, even if you are unsure.
We require this because we use automatic parsing."""

        elif task_name == "screen_gene_retrieval":
            # Screen gene retrieval - expect a gene symbol
            return """

You MUST include your answer (the gene symbol) within the following tags:
[ANSWER] and [/ANSWER]. For example, '[ANSWER]TP53[/ANSWER]',
where the answer is the gene symbol. Always answer in exactly this format
between the two tags, even if you are unsure.
We require this because we use automatic parsing."""

        else:
            # Default instruction for other tasks
            return """

You MUST include your answer within the following tags:
[ANSWER] and [/ANSWER]. For example, '[ANSWER]<your answer>[/ANSWER]'.
Always answer in exactly this format between the two tags, even if you are unsure.
We require this because we use automatic parsing."""

    def get_example(self, index=None):
        """
        Get a single example from the task

        Args:
            index: Index of the example (0-based). If None, returns a random example.

        Returns:
            dict: Dictionary with 'prompt', 'answer', 'task_instance_id', and 'task_name'
        """
        if index is None:
            index = np.random.randint(len(self.df))

        if index < 0 or index >= len(self.df):
            raise ValueError(
                f"Index {index} out of range for task {self.task_name} (max: {len(self.df)-1})"
            )

        row = self.df.iloc[index]

        # Append answer format instruction to the prompt
        prompt_with_instruction = row["prompt"] + self._answer_instruction

        return {
            "prompt": prompt_with_instruction,
            "answer": row["answer"],
            "task_instance_id": row["task_instance_id"],
            "task_name": row["task_name"],
            "split": row["split"],
        }

    def get_iterator(self):
        """Iterate over all examples in the task"""
        for i in range(len(self.df)):
            yield self.get_example(i)

    def evaluate(self, user_answer: str, task_instance_id: int):
        """
        Evaluate a user's answer for a specific instance

        Args:
            user_answer: User's answer
            task_instance_id: Task-specific instance ID

        Returns:
            float: Reward score (0.0 to 1.0)
        """
        return self.evaluator.evaluate(self.task_name, task_instance_id, user_answer)

    def __len__(self):
        return len(self.df)
