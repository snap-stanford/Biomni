from bioagentos.task.base_task import base_task
import pandas as pd
import numpy as np

class gwas_causal_gene(base_task):

    def __init__(self, dataset = 'opentargets', num_samples = 100):
        if dataset not in ['opentargets', 'gwas_catalog', 'pharmaprojects']:
            raise ValueError('dataset must be one of opentargets, gwas_catalog, pharmaprojects')
        
        query_path = '/dfs/user/kexinh/BioAgentOS/data/llm_gwas/data/benchmark_datasets/'  + dataset + '_step2.for_llm.tsv'
        answer_path = '/dfs/user/kexinh/BioAgentOS/data/llm_gwas/data/benchmark_datasets/'  + dataset + '_step2.labels'

        self.prompt = "Your task is to identify likely causal genes within a locus for a given GWAS phenotype. From the list, provide only the likely causal gene (matching one of the given genes). \nIdentify the causal gene.\nGWAS phenotype: {trait}\nGenes in locus: {gene_str}\n"
        self.query = pd.read_csv(query_path, sep = '\t').sample(frac = 1, random_state = 42).reset_index(drop=True)[:num_samples]
        self.answer = pd.read_csv(answer_path, sep = '\t').sample(frac = 1, random_state = 42).reset_index(drop=True)[:num_samples]

    def __len__(self):
        return len(self.query)

    def get_example(self, index = None):
        if index is None:
            q = self.query.sample(n=1).iloc[0]
        else:
            q = self.query.iloc[index]
        
        return {"prompt": self.prompt.format(trait = q.description, gene_str = q.symbol_gene_string), 
                "answer": self.answer.iloc[index].symbol}

    def split(self, ratio = 0.8, seed = 42):
        np.random.seed(seed)
        indices = np.arange(len(self.query))
        np.random.shuffle(indices)
        split_idx = int(ratio * len(self.query))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        return train_indices, val_indices

    def reward(self, input, output):
        answer = self.get_example(input)['answer']
        return 1 if output == answer else 0

    def get_iterator(self):
        for i in range(len(self.query)):
            yield self.get_example(i)

    def evaluate(self, response):
        ## expected a list/array of symbols
        from sklearn.metrics import accuracy_score
        predicted = [i.strip('{}') for i in response]
        ground_truth = self.answer['symbol'].values
        accuracy = accuracy_score(ground_truth, predicted)

        return {
            'accuracy': accuracy
        }

    def output_class(self):
        from pydantic import BaseModel, Field
        from typing import Optional
        class GeneOutput(BaseModel):
            """Causal gene output."""

            causal_gene: Optional[str] = Field(
                description="causal gene in the format of {gene_name}, e.g. {BRCA1}"
            )
        return GeneOutput

    def output_parser(self, output, llm):
        from pydantic import BaseModel, Field
        from typing import Optional
        class GeneOutput(BaseModel):
            """Causal gene output."""

            causal_gene: Optional[str] = Field(
                description="causal gene in the format of {gene_name}, e.g. {BRCA1}"
            )

        output_parser = llm.with_structured_output(GeneOutput)
        output = output_parser.invoke(output)
        return output.causal_gene
