import numpy as np
import pandas as pd
from bioagentos.task.base_task import base_task


class gwas_variant_prioritization(base_task):
    def __init__(self, num_samples=100):
        self.df = pd.read_pickle("/dfs/user/kexinh/BioAgentOS/data/gwas_gold_standards_benchmark.pkl")
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)[:num_samples]

        self.prompt = "Your task is to identify the most promising variant associated wtih a given GWAS phenotype for futher examination. \nFrom the list, prioritize the top associated variant (matching one of the given variant). \nGWAS phenotype: {trait}\nVariants: {variant_list}\n"
        self.num_examples = len(self.df)
        self.answer = self.df.rsid.values

    def __len__(self):
        return self.num_examples

    def get_example(self, index=None):
        if index is None:
            index = np.random.randint(self.num_examples)
        q = self.df.iloc[index]
        trait = q.trait_name

        total_list = [q.rsid] + q.random_rsids
        np.random.seed(index)
        np.random.shuffle(total_list)

        return {"prompt": self.prompt.format(trait=trait, variant_list=", ".join(total_list)), "answer": q.rsid}

    def split(self, ratio=0.8, seed=42):
        np.random.seed(seed)
        indices = np.arange(self.num_examples)
        np.random.shuffle(indices)
        split_idx = int(ratio * self.num_examples)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        return train_indices, val_indices

    def reward(self, input, output):
        answer = self.get_example(input)["answer"]
        return 1 if output == answer else 0

    def get_iterator(self):
        for i in range(len(self.query)):
            yield self.get_example(i)

    def evaluate(self, response):
        ## expected a list/array of symbols
        from sklearn.metrics import accuracy_score

        predicted = [i.strip("{}") for i in response]
        ground_truth = self.answer
        accuracy = accuracy_score(ground_truth, predicted)

        total_chisq = []
        miss_num = 0
        for idx, i in enumerate(predicted):
            if i in self.df.ID.values:
                tmp_df = self.df[self.df["index"] == idx]
                total_chisq.append(tmp_df[tmp_df.ID == predicted].CHISQ.values[0])
            else:
                miss_num += 1

        return {"accuracy": accuracy, "miss_num": miss_num, "mean_chisq": np.mean(total_chisq)}

    def output_class(self):
        from typing import Optional

        from pydantic import BaseModel, Field

        class VariantOutput(BaseModel):
            """Prioritized variant output."""

            variant: str | None = Field(description="variant ID in the format of {variant_ID}, e.g. {2_57743808_G_A}")

        return VariantOutput
