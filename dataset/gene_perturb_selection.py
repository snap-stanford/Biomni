from bioagentos.task.base_task import base_task
import json, pandas as pd, numpy as np

class gene_perturb_selection(base_task):
    def __init__(self, dataset='IL2',
                num_query_genes=10):
        json_file_path = '/dfs/user/yhr/AI_RA/research_assistant/datasets/task_prompts/' + dataset + '.json'
        with open(json_file_path, 'r') as f:
            prompt_data = json.load(f)

        self.task_description = prompt_data['Task']
        
        ground_truth_path = f'/dfs/user/yhr/AI_RA/research_assistant/datasets/ground_truth_{dataset}.csv'
        hit_genes_path = f'/dfs/user/yhr/AI_RA/research_assistant/datasets/topmovers_{dataset}.npy'

        self.ground_truth = pd.read_csv(ground_truth_path, index_col=0)
        self.all_hit_genes = np.load(hit_genes_path)

        self.query = []
        self.answer = []
        np.random.seed(42)
        non_hit_genes = np.setdiff1d(self.ground_truth.index.values, self.all_hit_genes)
        for hit in self.all_hit_genes:
            sampled_non_hit_genes = np.random.choice(non_hit_genes, num_query_genes-1, replace=False).tolist()
            sampled_non_hit_genes += [hit]
            np.random.shuffle(sampled_non_hit_genes)
            self.query.append(','.join(sampled_non_hit_genes))
            self.answer.append(hit)

        self.prompt = "Your task is to {task_description}. \n From the list of potential genes, provide one most confident gene (matching one of the given genes). \n Gene list: {gene_list}"

    def get_example(self, index = None):
        if index is None:
            index = np.random.randint(len(self.query))
        
        q = self.query[index]
        a = self.answer[index]
        return {"prompt": self.prompt.format(task_description = self.task_description, gene_list = q), 
                "answer": a}

    def reward(self, input, output):
        answer = self.get_example(input)['answer']
        return 1 if output == answer else 0

    def split(self, ratio = 0.8, seed = 42):
        np.random.seed(seed)
        indices = np.arange(len(self.query))
        np.random.shuffle(indices)
        split_idx = int(ratio * len(self.query))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        return train_indices, val_indices
    
    def get_iterator(self):
        for i in range(len(self.query)):
            yield self.get_example(i)

    def output_class(self):
        from pydantic import BaseModel, Field
        from typing import Optional
        class GeneOutput(BaseModel):
            """A selected gene to conduct perturbation."""

            gene_name: Optional[str] = Field(
                description="A selected gene to conduct perturbation., e.g. {BRCA1}"
            )
        return GeneOutput

    def evaluate(self, response):
        ## expected a list/array of symbols
        from sklearn.metrics import accuracy_score
        predicted = [i.strip('{}') for i in response]
        ground_truth = self.answer
        return {
            'accuracy': accuracy_score(ground_truth, predicted),
            'miss_num': len(np.setdiff1d(predicted, self.ground_truth.index.values)),
            'average_absolute_perturbation_effect': self.ground_truth.loc[np.intersect1d(predicted, self.ground_truth.index.values)].Score.abs().mean(),
            'hit_ratio': np.mean([1 if gene in self.all_hit_genes else 0 for gene in predicted])
        }