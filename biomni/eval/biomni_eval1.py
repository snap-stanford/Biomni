"""
BiomniEval1: Evaluation loader for Biomni tasks

This class provides a unified interface to evaluate user answers against ground truth
for all tasks in the BiomniEval1 benchmark.
"""

import json
from typing import Any

import pandas as pd


class BiomniEval1:
    """
    Evaluation loader for BiomniEval1 benchmark

    Usage:
        evaluator = BiomniEval1('biomni_eval1_dataset.parquet')
        score = evaluator.evaluate('gwas_causal_gene_opentargets', 0, 'BRCA1')
    """

    def __init__(self):
        """
        Initialize the BiomniEval1 evaluator

        Args:
            dataset_path: Path to the merged dataset parquet file
        """

        self.df = pd.read_parquet("hf://datasets/biomni/Eval1/biomni_eval1_dataset.parquet")

        # Create index mapping for fast lookup using task_instance_id
        self.instance_map = {}
        for idx, row in self.df.iterrows():
            key = (row["task_name"], row["task_instance_id"])
            self.instance_map[key] = idx

        print(f"Loaded BiomniEval1 dataset: {len(self.df)} instances across {self.df['task_name'].nunique()} tasks")

    def evaluate(self, task_name: str, task_instance_id: int, user_answer: str) -> float:
        """
        Evaluate a user's answer for a given task and instance

        Args:
            task_name: Name of the task (e.g., 'gwas_causal_gene_opentargets')
            task_instance_id: Task-specific instance ID (not the global instance_id)
            user_answer: User's answer (format depends on task)

        Returns:
            float: Reward score (0.0 to 1.0)
        """
        # Look up the instance in the dataset using task_instance_id
        key = (task_name, task_instance_id)
        if key not in self.instance_map:
            raise ValueError(f"Instance not found: task={task_name}, task_instance_id={task_instance_id}")

        df_idx = self.instance_map[key]
        row = self.df.iloc[df_idx]
        ground_truth = row["answer"]

        # Call task-specific evaluation function
        try:
            reward = self._compute_reward(task_name, user_answer, ground_truth)
            return float(reward)

        except Exception as e:
            raise RuntimeError(f"Error computing reward for {task_name} instance {task_instance_id}: {e}")

    def _compute_reward(self, task_name: str, user_answer: str, ground_truth: str) -> float:
        """Compute reward using task-specific logic"""

        if task_name == "crispr_delivery":
            # CRISPR expects answer as a letter (a-f), exact match
            return 1.0 if user_answer.strip().lower() == ground_truth.strip().lower() else 0.0

        elif task_name.startswith("gwas_causal_gene"):
            # GWAS causal gene expects exact gene match (case-insensitive)
            return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0

        elif task_name == "gwas_variant_prioritization":
            # GWAS variant expects exact variant match
            return 1.0 if user_answer.strip() == ground_truth.strip() else 0.0

        elif task_name == "hle":
            # HLE expects letter answer (A-Z), case-insensitive
            return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0

        elif task_name.startswith("lab_bench"):
            # Lab bench expects letter answer (A-Z), case-insensitive
            return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0

        elif task_name == "rare_disease_diagnosis":
            # Rare disease expects JSON with OMIM_ID match
            # Parse both user answer and ground truth
            try:
                if isinstance(user_answer, str):
                    try:
                        user_dict = json.loads(user_answer)
                    except json.JSONDecodeError:
                        import ast

                        user_dict = ast.literal_eval(user_answer)
                else:
                    user_dict = user_answer

                if isinstance(ground_truth, str):
                    gt_dict = json.loads(ground_truth)
                else:
                    gt_dict = ground_truth

                # Compare OMIM_ID
                return 1.0 if user_dict.get("OMIM_ID") == gt_dict.get("OMIM_ID") else 0.0

            except Exception:
                return 0.0

        elif task_name == "screen_gene_retrieval":
            # Screen gene retrieval expects gene symbol (case-insensitive)
            return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0

        elif task_name == "patient_gene_detection":
            # Patient gene detection expects JSON with causal_gene list
            # Ground truth is a comma-separated string or single gene ID
            try:
                if isinstance(user_answer, str):
                    try:
                        user_dict = json.loads(user_answer)
                    except json.JSONDecodeError:
                        import ast

                        user_dict = ast.literal_eval(user_answer)
                else:
                    user_dict = user_answer

                # Get predicted genes
                predicted_genes = user_dict.get("causal_gene", [])
                if not isinstance(predicted_genes, list):
                    predicted_genes = [predicted_genes]

                # Get ground truth genes (stored as comma-separated or single)
                if "," in ground_truth:
                    true_genes = [g.strip() for g in ground_truth.split(",")]
                else:
                    true_genes = [ground_truth]

                # Check for intersection
                if predicted_genes and set(true_genes) & set(predicted_genes):
                    return 1.0
                else:
                    return 0.0

            except Exception:
                return 0.0

        else:
            raise ValueError(f"Unknown task: {task_name}")

    def get_instance(self, task_name: str, task_instance_id: int) -> dict[str, Any]:
        """
        Get information about a specific instance

        Args:
            task_name: Name of the task
            task_instance_id: Task-specific instance ID

        Returns:
            dict: Instance information including prompt, answer, etc.
        """
        key = (task_name, task_instance_id)
        if key not in self.instance_map:
            raise ValueError(f"Instance not found: task={task_name}, task_instance_id={task_instance_id}")

        df_idx = self.instance_map[key]
        row = self.df.iloc[df_idx]

        return {
            "global_instance_id": row["instance_id"],
            "task_instance_id": row["task_instance_id"],
            "task_name": row["task_name"],
            "split": row["split"],
            "prompt": row["prompt"],
            "answer": row["answer"],
        }

    def list_tasks(self) -> list:
        """Get list of all available tasks"""
        return sorted(self.df["task_name"].unique().tolist())

    def get_task_stats(self, task_name: str = None) -> dict[str, Any]:
        """
        Get statistics for a task or all tasks

        Args:
            task_name: Optional task name to filter by

        Returns:
            dict: Statistics including counts by split
        """
        if task_name:
            task_df = self.df[self.df["task_name"] == task_name]
            if len(task_df) == 0:
                raise ValueError(f"Task not found: {task_name}")
        else:
            task_df = self.df

        stats = {
            "total_instances": len(task_df),
            "train_instances": len(task_df[task_df["split"] == "train"]),
            "val_instances": len(task_df[task_df["split"] == "val"]),
        }

        if not task_name:
            stats["tasks"] = {}
            for tn in self.list_tasks():
                stats["tasks"][tn] = self.get_task_stats(tn)

        return stats

    def batch_evaluate(self, evaluations: list) -> list:
        """
        Evaluate multiple instances at once

        Args:
            evaluations: List of tuples (task_name, task_instance_id, user_answer)

        Returns:
            list: List of reward scores
        """
        results = []
        for task_name, task_instance_id, user_answer in evaluations:
            try:
                score = self.evaluate(task_name, task_instance_id, user_answer)
                results.append(score)
            except Exception as e:
                print(f"Error evaluating {task_name} instance {task_instance_id}: {e}")
                results.append(0.0)

        return results

    def get_instances_by_task(self, task_name: str, split: str = None) -> pd.DataFrame:
        """
        Get all instances for a specific task

        Args:
            task_name: Name of the task
            split: Optional split filter ('train' or 'val')

        Returns:
            DataFrame with instances
        """
        task_df = self.df[self.df["task_name"] == task_name]

        if split:
            task_df = task_df[task_df["split"] == split]

        return task_df.copy()

    def __repr__(self):
        return f"BiomniEval1(instances={len(self.df)}, tasks={self.df['task_name'].nunique()})"

    def __len__(self):
        return len(self.df)


def main():
    """Demo usage of BiomniEval1"""
    evaluator = BiomniEval1()

    print("\nAvailable tasks:")
    for task in evaluator.list_tasks():
        print(f"  - {task}")

    print("\nOverall statistics:")
    stats = evaluator.get_task_stats()
    print(f"  Total instances: {stats['total_instances']}")
    print(f"  Train: {stats['train_instances']}, Val: {stats['val_instances']}")

    print("\nPer-task statistics:")
    for task_name in evaluator.list_tasks():
        task_stats = evaluator.get_task_stats(task_name)
        print(
            f"  {task_name}: {task_stats['total_instances']} total ({task_stats['train_instances']} train, {task_stats['val_instances']} val)"
        )

    # Example evaluation
    print("\n" + "=" * 60)
    print("Example evaluation:")
    print("=" * 60)

    # Get first instance from gwas_variant_prioritization
    first_instance = evaluator.df[evaluator.df["task_name"] == "gwas_variant_prioritization"].iloc[0]
    task_name = first_instance["task_name"]
    task_instance_id = first_instance["task_instance_id"]
    ground_truth = first_instance["answer"]

    print(f"\nTask: {task_name}")
    print(f"Task Instance ID: {task_instance_id}")
    print(f"Ground truth: {ground_truth}")
    print(f"Prompt preview: {first_instance['prompt'][:200]}...")

    # Test with correct answer
    score = evaluator.evaluate(task_name, task_instance_id, ground_truth)
    print(f"\nScore (correct answer '{ground_truth}'): {score}")

    # Test with wrong answer
    score = evaluator.evaluate(task_name, task_instance_id, "wrong_answer")
    print(f"Score (wrong answer 'wrong_answer'): {score}")

    # Batch evaluation example
    print("\n" + "=" * 60)
    print("Batch evaluation example:")
    print("=" * 60)
    batch_evals = [
        (task_name, task_instance_id, ground_truth),  # Correct
        (task_name, task_instance_id, "wrong"),  # Wrong
    ]
    scores = evaluator.batch_evaluate(batch_evals)
    print(f"Batch scores: {scores}")


if __name__ == "__main__":
    main()
