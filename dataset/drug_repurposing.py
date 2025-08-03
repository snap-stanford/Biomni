import numpy as np
import pandas as pd
from bioagentos.task.base_task import base_task


class drug_repurposing(base_task):
    def __init__(self, num_samples=100):
        # Load the benchmark dataset
        self.df = pd.read_csv("/dfs/user/kexinh/BioAgentOS/data/drug_repurposing_ehr_validation.csv")
        self.df = self.df[self.df.indicated != 1]
        # Define the prompt template
        self.prompt = (
            "Your task is to identify top 5 drugs that can be potentially repurposed to treat the given disease. "
            "From the list, prioritize the drug list with the highest potential (matching the given DrugBank IDs).\n"
            "Disease: {disease}\nDrugs: {drug_list}\n"
            "Output format: a list of drugs with their DrugBank IDs, no drug name, just the IDs: 1. DB00001 2. DB00002 3. DB00003 .."
        )
        self.disease_names = (
            self.df.groupby("disease_name").log_OR.mean().sort_values()[::-1][:num_samples].index.values
        )
        self.num_examples = len(self.disease_names)

    def __len__(self):
        return self.num_examples

    def get_example(self, index=None):
        if index is None:
            index = np.random.randint(self.num_examples)
        example = self.disease_names[index]
        novel_db_ids = self.df[self.df.disease_name == example].sort_values("log_OR")[::-1].DB_ID.values[:50]

        return {"prompt": self.prompt.format(disease=example, drug_list=", ".join(novel_db_ids))}

    def get_iterator(self):
        for i in range(self.num_examples):
            yield self.get_example(i)

    def evaluate(self, disease_name, drugs):
        # Evaluation using accuracy
        out = self.df[self.df.disease_name == disease_name].set_index("DB_ID").loc[drugs].log_OR

        return {"mean_log_or": out.mean(), "max_log_or": out.max()}

    def reward(self, input, output):
        name = self.disease_names[input]
        return self.df[self.df.disease_name == name].set_index("DB_ID").loc[output].log_OR.mean()

    def split(self, ratio=0.8, seed=42):
        np.random.seed(seed)
        indices = np.arange(self.num_examples)
        np.random.shuffle(indices)
        split_idx = int(ratio * self.num_examples)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        return train_indices, val_indices

    def output_class(self):
        from typing import Optional

        from pydantic import BaseModel, Field

        class drug_list(BaseModel):
            """List of drugs to repurpose for a disease"""

            drug_list: str = Field(
                description="""A list of drug bank IDs to repurpose for a disease.
                The drugs should be separated by a newline character. The output should be DB00001\nDB00002\nDB00003\n...\nDB00005"""
            )

        return drug_list
