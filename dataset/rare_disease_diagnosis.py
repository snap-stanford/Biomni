import json
import sys

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, Field

sys.path.append("/dfs/user/kexinh/BioAgentOS")
from bioagentos.llm import get_llm
from bioagentos.task.base_task import base_task


def get_gene_name_from_ensembl(ensembl_id):
    """
    Retrieve the gene name associated with a given Ensembl ID using MyGene.info API.

    Args:
        ensembl_id (str): The Ensembl ID to query.

    Returns:
        str: The gene name if found, or a message indicating it wasn't found.
    """
    base_url = "https://mygene.info/v3/query"
    params = {"q": ensembl_id, "fields": "symbol", "species": "human"}  # Update species as needed

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an error for HTTP issues
        data = response.json()

        if "hits" in data and len(data["hits"]) > 0:
            return ensembl_id[0] + ": " + data["hits"][0].get("symbol", ensembl_id)
        else:
            return ensembl_id
    except requests.exceptions.RequestException:
        return ensembl_id


def parse_hpo_obo(file_path):
    """
    Parse the HPO OBO file and create a dictionary mapping HP IDs to phenotype descriptions.

    Args:
        file_path (str): Path to the HPO OBO file.

    Returns:
        dict: A dictionary where keys are HP IDs and values are phenotype descriptions.
    """
    hp_dict = {}
    current_id = None
    current_name = None

    with open(file_path) as file:
        for line in file:
            line = line.strip()
            if line.startswith("[Term]"):
                # If a new term block starts, save the previous term
                if current_id and current_name:
                    hp_dict[current_id] = current_name
                current_id = None
                current_name = None
            elif line.startswith("id: HP:"):
                current_id = line.split(": ")[1]
            elif line.startswith("name:"):
                current_name = line.split(": ", 1)[1]

        # Add the last term to the dictionary
        if current_id and current_name:
            hp_dict[current_id] = current_name

    return hp_dict


# Example usage
file_path = "/dfs/user/kexinh/BioAgentOS/data/hp.obo"  # Replace with the path to your hp.obo file
hp_dict = parse_hpo_obo(file_path)


class rare_disease_diagnosis(base_task):
    def __init__(self, eval_llm="claude-3-5-sonnet-20241022", num_samples=None, map_id=False):
        self.map_id = map_id
        self.llm = get_llm(eval_llm)
        data_path = "/dfs/user/kexinh/BioAgentOS/data/mygene.json"
        data = []
        with open(data_path) as file:
            for line in file:
                data.append(json.loads(line))

        if num_samples is None:
            self.data = pd.DataFrame(data)
        else:
            self.data = pd.DataFrame(data)[:num_samples]

        # Ensure the data contains all necessary columns
        required_columns = ["id", "positive_phenotypes", "all_candidate_genes", "omim", "disease_name", "orpha_id"]
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Dataset is missing required column: {col}")

        self.query = []
        self.answer = []
        for _, row in self.data.iterrows():
            phenotypes = row["positive_phenotypes"]
            candidate_genes = row["all_candidate_genes"]
            disease_name = row["disease_name"]
            omim_id = row["omim"]

            self.query.append({"phenotypes": phenotypes, "candidate_genes": candidate_genes})
            self.answer.append({"disease_name": disease_name, "OMIM_ID": omim_id})

        self.task_description = """
Task: given a patient's phenotypes and a list of candidate genes, diagnose the rare disease that the patient has.
Phenotypes: {phenotype_list}
Candidate genes: {candidate_genes}

Output format: {{'disease_name': XXX, 'OMIM_ID': XXX}}
        """

        self.completion_checker = """
Given an answer and a solution, check if the answer is correct.

Answer: {answer}
Solution: {solution}

Return 'task completed' if the answer is correct, and 'task not completed' otherwise.
        """

    def __len__(self):
        return len(self.query)

    def get_example(self, index=None):
        if index is None:
            index = np.random.randint(len(self.query))

        q = self.query[index]
        a = self.answer[index]

        if self.map_id:
            prompt = self.task_description.format(
                phenotype_list=", ".join([hp_dict.get(hp_id, hp_id) for hp_id in q["phenotypes"]]),
                candidate_genes=get_gene_name_from_ensembl(q["candidate_genes"]),
            )
        else:
            prompt = self.task_description.format(
                phenotype_list=", ".join(q["phenotypes"]), candidate_genes=q["candidate_genes"]
            )

        return {"prompt": prompt, "answer": a}

    def split(self, ratio=0.8, seed=42):
        np.random.seed(seed)
        indices = np.arange(len(self.query))
        np.random.shuffle(indices)
        split_idx = int(ratio * len(self.query))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        return train_indices, val_indices

    def reward(self, input, output):
        answer = self.get_example(input)["answer"]
        return 1 if output.OMIM_ID == answer.OMIM_ID else 0

    def get_iterator(self):
        for i in range(len(self.query)):
            yield self.get_example(i)

    def output_class(self):
        from typing import Optional

        from pydantic import BaseModel, Field

        class DiagnosisOutput(BaseModel):
            """A diagnosis for a rare disease."""

            disease_name: str | None = Field(
                description="The name of the diagnosed rare disease, e.g., 'Marfan Syndrome'"
            )
            OMIM_ID: str | None = Field(description="The OMIM ID of the diagnosed disease, e.g., '154700'")

        return DiagnosisOutput

    def evaluate(self, response, ground_truth=None):
        from sklearn.metrics import accuracy_score

        if ground_truth is None:
            ground_truth = self.answer
        predicted = response
        correct = []
        results = []
        for pred, gt in zip(predicted, ground_truth, strict=False):
            # Use the LLM-based completion checker to verify each prediction
            check_prompt = self.completion_checker.format(answer=json.dumps(pred), solution=json.dumps(gt))
            # Assuming an LLM API call here; replace with the actual implementation
            result = self.call_llm_to_check(check_prompt)
            correct.append(result == "task completed")
            results.append(result)

        accuracy = accuracy_score([1] * len(correct), correct)
        return {
            "completion_rate": accuracy,
            "num_of_tasks_completed": sum(correct),
            "num_of_total_tasks": len(correct),
            "results": results,
        }

    def call_llm_to_check(self, prompt):
        class output_format(BaseModel):
            """Parse if the task is completed or not."""

            completion_status: str = Field(
                description="""'task completed' if the answer shows that it is completed, and 'task not completed' otherwise."""
            )

        self.llm.with_structured_output(output_format)
        return self.llm.invoke(prompt).completion_status
