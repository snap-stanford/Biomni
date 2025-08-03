import sys

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

sys.path.append("/dfs/user/kexinh/BioAgentOS")
from bioagentos.task.base_task import base_task


class patient_gene_detection(base_task):
    def __init__(self, num_samples=100, map_id=True):
        self.map_id = map_id
        if self.map_id:
            self.data = pd.read_pickle("/dfs/user/kexinh/BioAgentOS/data/patient_gene_detection_benchmark_mapped.pkl")
        else:
            self.data = pd.read_pickle("/dfs/user/kexinh/BioAgentOS/data/patient_gene_detection_benchmark_unmapped.pkl")

        self.query = []
        self.answer = []
        self.data = self.data[:num_samples]
        for idx in range(len(self.data)):
            patient = self.data.iloc[idx]

            phenotypes = patient["phenotypes"]
            candidate_genes = patient["candidate_genes"]
            true_genes = patient["true_genes"]

            self.query.append({"phenotypes": phenotypes, "candidate_genes": candidate_genes})
            self.answer.append({"true_genes": true_genes})

        self.task_description = """
Task: Given a patient's phenotypes and a list of candidate genes, identify the causal gene.
Phenotypes: {phenotype_list}
Candidate genes: {candidate_genes}

Output format: {{'causal_gene': [gene1]}}
        """

    def __len__(self):
        return len(self.query)

    def get_example(self, index=None):
        if index is None:
            index = np.random.randint(len(self.query))

        q = self.query[index]
        a = self.answer[index]

        prompt = self.task_description.format(
            phenotype_list=", ".join(q["phenotypes"]), candidate_genes=", ".join(q["candidate_genes"])
        )
        answer = a["true_genes"]

        return {"prompt": prompt, "answer": answer}

    def get_iterator(self):
        for i in range(len(self.query)):
            yield self.get_example(i)

    def split(self, ratio=0.8, seed=42):
        np.random.seed(seed)
        indices = np.arange(len(self.query))
        np.random.shuffle(indices)
        split_idx = int(ratio * len(self.query))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        return train_indices, val_indices

    def reward(self, input, output):
        true_genes = self.get_example(input)["answer"]
        if np.intersect1d(true_genes, output):
            return 1.0
        else:
            return 0.0

    def output_class(self):
        class GeneDetectionOutput(BaseModel):
            """List of predicted causal genes for a patient."""

            causal_genes: list = Field(
                description="The list of causal gene(s) identified, e.g., ['ENSG00000138449']. Please Use ENSG ID."
            )

        return GeneDetectionOutput
