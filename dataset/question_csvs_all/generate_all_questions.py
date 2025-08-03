import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path

# Add the biomni directory to the path
sys.path.append('biomni')

# Simple base task class
class base_task:
    def __init__(self):
        pass
    def get_example(self):
        pass
    def get_iterator(self):
        pass
    def evaluate(self):
        pass
    def output_class(self):
        pass
    def get_prompt_from_input(self, input):
        return self.get_example(input)['prompt']

def create_output_directory():
    """Create output directory for CSV files"""
    output_dir = Path("question_csvs_all")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def generate_drug_repurposing_questions(output_dir):
    """Generate all questions for drug repurposing task"""
    print("Generating drug repurposing questions...")
    
    try:
        # Load the benchmark dataset from local path
        df = pd.read_csv('dataset/drug_repurposing/drug_repurposing_ehr_validation.csv')
        df = df[df.indicated != 1]
        
        # Get all unique disease names
        disease_names = df.groupby('disease_name').log_OR.mean().sort_values()[::-1].index.values
        
        prompt_template = (
            "Your task is to identify top 5 drugs that can be potentially repurposed to treat the given disease. "
            "From the list, prioritize the drug list with the highest potential (matching the given DrugBank IDs).\n"
            "Disease: {disease}\nDrugs: {drug_list}\n"
            "Output format: a list of drugs with their DrugBank IDs, no drug name, just the IDs: 1. DB00001 2. DB00002 3. DB00003 .."
        )
        
        questions = []
        for disease in disease_names:
            novel_db_ids = df[df.disease_name == disease].sort_values('log_OR')[::-1].DB_ID.values[:50]
            
            # Get the top 5 DrugBank IDs as the actual answer
            top_5_ids = novel_db_ids[:5]
            answer = ", ".join([f"{i+1}. {id}" for i, id in enumerate(top_5_ids)])
            
            prompt = prompt_template.format(
                disease=disease, 
                drug_list=", ".join(novel_db_ids)
            )
            
            questions.append({
                'task': 'drug_repurposing',
                'prompt': prompt,
                'answer': answer
            })
        
        df_out = pd.DataFrame(questions)
        df_out.to_csv(output_dir / 'drug_repurposing_questions.csv', index=False)
        print(f"Generated {len(df_out)} drug repurposing questions")
        return df_out
    except Exception as e:
        print(f"Error generating drug repurposing questions: {e}")
        return pd.DataFrame()

def generate_gene_perturb_selection_questions(output_dir):
    """Generate all questions for gene perturbation selection task"""
    print("Generating gene perturbation selection questions...")
    
    try:
        # Load data
        json_file_path = 'dataset/gene_perturb_selection/IL2.json'
        with open(json_file_path, 'r') as f:
            prompt_data = json.load(f)

        task_description = prompt_data['Task']
        
        ground_truth_path = 'dataset/gene_perturb_selection/ground_truth_IL2.csv'
        hit_genes_path = 'dataset/gene_perturb_selection/topmovers_IL2.npy'

        ground_truth = pd.read_csv(ground_truth_path, index_col=0)
        all_hit_genes = np.load(hit_genes_path)

        query = []
        answer = []
        np.random.seed(42)
        non_hit_genes = np.setdiff1d(ground_truth.index.values, all_hit_genes)
        
        # Generate questions for each hit gene
        for hit in all_hit_genes:
            sampled_non_hit_genes = np.random.choice(non_hit_genes, 9, replace=False).tolist()
            sampled_non_hit_genes += [hit]
            np.random.shuffle(sampled_non_hit_genes)
            query.append(','.join(sampled_non_hit_genes))
            answer.append(hit)

        prompt_template = "Your task is to {task_description}. \n From the list of potential genes, provide one most confident gene (matching one of the given genes). \n Gene list: {gene_list}"
        
        questions = []
        for i in range(len(query)):
            prompt = prompt_template.format(task_description=task_description, gene_list=query[i])
            questions.append({
                'task': 'gene_perturb_selection',
                'prompt': prompt,
                'answer': answer[i]
            })
        
        df_out = pd.DataFrame(questions)
        df_out.to_csv(output_dir / 'gene_perturb_selection_questions.csv', index=False)
        print(f"Generated {len(df_out)} gene perturbation selection questions")
        return df_out
    except Exception as e:
        print(f"Error generating gene perturbation selection questions: {e}")
        return pd.DataFrame()

def generate_gwas_causal_gene_questions(output_dir):
    """Generate all questions for GWAS causal gene task"""
    print("Generating GWAS causal gene questions...")
    
    all_questions = []
    
    for dataset in ['opentargets', 'gwas_catalog', 'pharmaprojects']:
        try:
            query_path = f'dataset/gwas_causal_gene/{dataset}_step2.for_llm.tsv'
            answer_path = f'dataset/gwas_causal_gene/{dataset}_step2.labels'

            prompt_template = "Your task is to identify likely causal genes within a locus for a given GWAS phenotype. From the list, provide only the likely causal gene (matching one of the given genes). \nIdentify the causal gene.\nGWAS phenotype: {trait}\nGenes in locus: {gene_str}\n"
            
            query_df = pd.read_csv(query_path, sep='\t')
            answer_df = pd.read_csv(answer_path, sep='\t')

            for i in range(len(query_df)):
                q = query_df.iloc[i]
                prompt = prompt_template.format(trait=q.description, gene_str=q.symbol_gene_string)
                answer = answer_df.iloc[i].symbol
                
                all_questions.append({
                    'task': f'gwas_causal_gene_{dataset}',
                    'prompt': prompt,
                    'answer': answer
                })
            print(f"Generated {len(query_df)} questions for {dataset}")
        except Exception as e:
            print(f"Error with dataset {dataset}: {e}")
    
    df_out = pd.DataFrame(all_questions)
    df_out.to_csv(output_dir / 'gwas_causal_gene_questions.csv', index=False)
    print(f"Generated {len(df_out)} total GWAS causal gene questions")
    return df_out

def generate_gwas_variant_prioritization_questions(output_dir):
    """Generate all questions for GWAS variant prioritization task"""
    print("Generating GWAS variant prioritization questions...")
    
    try:
        df = pd.read_pickle('dataset/gwas_variant_prioritization/gwas_gold_standards_benchmark.pkl')

        prompt_template = "Your task is to identify the most promising variant associated wtih a given GWAS phenotype for futher examination. \nFrom the list, prioritize the top associated variant (matching one of the given variant). \nGWAS phenotype: {trait}\nVariants: {variant_list}\n"
        
        questions = []
        for i in range(len(df)):
            q = df.iloc[i]
            trait = q.trait_name

            total_list = [q.rsid] + q.random_rsids
            np.random.seed(i)
            np.random.shuffle(total_list)

            prompt = prompt_template.format(trait=trait, variant_list=', '.join(total_list))
            
            questions.append({
                'task': 'gwas_variant_prioritization',
                'prompt': prompt,
                'answer': q.rsid
            })
        
        df_out = pd.DataFrame(questions)
        df_out.to_csv(output_dir / 'gwas_variant_prioritization_questions.csv', index=False)
        print(f"Generated {len(df_out)} GWAS variant prioritization questions")
        return df_out
    except Exception as e:
        print(f"Error generating GWAS variant prioritization questions: {e}")
        return pd.DataFrame()

def generate_hle_questions(output_dir):
    """Generate all questions for humanity last exam task"""
    print("Generating humanity last exam questions...")
    
    try:
        df = pd.read_parquet('dataset/hle/test_sampled_biology_medicine.parquet')

        # Filter for Biology/Medicine category and multiple choice questions
        df = df[df['category'] == 'Biology/Medicine']
        df = df[df['answer_type'] == 'multipleChoice']
        df['question_text'] = df.question
        df['letter_answer'] = df['answer'].apply(lambda x: x[0])

        prompt_template = """Question: {question}"""
        
        questions = []
        for _, row in df.iterrows():
            prompt = prompt_template.format(question=row['question_text'])
            questions.append({
                'task': 'humanity_last_exam',
                'prompt': prompt,
                'answer': row['letter_answer']
            })
        
        df_out = pd.DataFrame(questions)
        df_out.to_csv(output_dir / 'humanity_last_exam_questions.csv', index=False)
        print(f"Generated {len(df_out)} humanity last exam questions")
        return df_out
    except Exception as e:
        print(f"Error generating humanity last exam questions: {e}")
        return pd.DataFrame()

def generate_patient_gene_detection_questions(output_dir):
    """Generate all questions for patient gene detection task"""
    print("Generating patient gene detection questions...")
    
    try:
        data = pd.read_pickle('dataset/patient_gene_detection/patient_gene_detection_benchmark.pkl')

        task_description = """
Task: Given a patient's phenotypes and a list of candidate genes, identify the causal gene.
Phenotypes: {phenotype_list}
Candidate genes: {candidate_genes}

Output format: {{'causal_gene': [gene1]}}
        """
        
        questions = []
        for idx in range(len(data)):
            patient = data.iloc[idx]

            phenotypes = patient['phenotypes']
            candidate_genes = patient['candidate_genes']
            true_genes = patient['true_genes']

            prompt = task_description.format(
                phenotype_list=', '.join(phenotypes),
                candidate_genes=', '.join(candidate_genes)
            )
            
            questions.append({
                'task': 'patient_gene_detection',
                'prompt': prompt,
                'answer': str({'true_genes': true_genes})
            })
        
        df_out = pd.DataFrame(questions)
        df_out.to_csv(output_dir / 'patient_gene_detection_questions.csv', index=False)
        print(f"Generated {len(df_out)} patient gene detection questions")
        return df_out
    except Exception as e:
        print(f"Error generating patient gene detection questions: {e}")
        return pd.DataFrame()

def generate_rare_disease_diagnosis_questions(output_dir):
    """Generate all questions for rare disease diagnosis task"""
    print("Generating rare disease diagnosis questions...")
    
    try:
        data_path = 'dataset/rare_disease_diagnosis/mygene.json'
        data = []
        with open(data_path, "r") as file:
            for line in file:
                data.append(json.loads(line))

        df_data = pd.DataFrame(data)

        required_columns = ['id', 'positive_phenotypes', 'all_candidate_genes', 'omim', 'disease_name', 'orpha_id']
        for col in required_columns:
            if col not in df_data.columns:
                raise ValueError(f"Dataset is missing required column: {col}")

        task_description = """
Task: given a patient's phenotypes and a list of candidate genes, diagnose the rare disease that the patient has.
Phenotypes: {phenotype_list}
Candidate genes: {candidate_genes}

Output format: {{'disease_name': XXX, 'OMIM_ID': XXX}}
        """
        
        questions = []
        for _, row in df_data.iterrows():
            phenotypes = row['positive_phenotypes']
            candidate_genes = row['all_candidate_genes']
            disease_name = row['disease_name']
            omim_id = row['omim']

            prompt = task_description.format(
                phenotype_list=', '.join(phenotypes),
                candidate_genes=candidate_genes
            )
            
            questions.append({
                'task': 'rare_disease_diagnosis',
                'prompt': prompt,
                'answer': str({'disease_name': disease_name, 'OMIM_ID': omim_id})
            })
        
        df_out = pd.DataFrame(questions)
        df_out.to_csv(output_dir / 'rare_disease_diagnosis_questions.csv', index=False)
        print(f"Generated {len(df_out)} rare disease diagnosis questions")
        return df_out
    except Exception as e:
        print(f"Error generating rare disease diagnosis questions: {e}")
        return pd.DataFrame()

def main():
    """Main function to generate all question CSV files"""
    output_dir = create_output_directory()
    
    all_dataframes = []
    
    # Generate questions for each task
    df1 = generate_drug_repurposing_questions(output_dir)
    if not df1.empty:
        all_dataframes.append(df1)
    
    df2 = generate_gene_perturb_selection_questions(output_dir)
    if not df2.empty:
        all_dataframes.append(df2)
    
    df3 = generate_gwas_causal_gene_questions(output_dir)
    if not df3.empty:
        all_dataframes.append(df3)
    
    df4 = generate_gwas_variant_prioritization_questions(output_dir)
    if not df4.empty:
        all_dataframes.append(df4)
    
    df5 = generate_hle_questions(output_dir)
    if not df5.empty:
        all_dataframes.append(df5)
    
    df6 = generate_patient_gene_detection_questions(output_dir)
    if not df6.empty:
        all_dataframes.append(df6)
    
    df7 = generate_rare_disease_diagnosis_questions(output_dir)
    if not df7.empty:
        all_dataframes.append(df7)
    
    # Combine all dataframes
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df.to_csv(output_dir / 'all_tasks_combined.csv', index=False)
        print(f"\nGenerated combined CSV with {len(combined_df)} total questions")
        print(f"Output directory: {output_dir}")
        
        # Print summary
        print("\nSummary of generated questions:")
        for df in all_dataframes:
            task_name = df['task'].iloc[0] if not df.empty else "Unknown"
            print(f"- {task_name}: {len(df)} questions")
    else:
        print("No questions were generated successfully")

if __name__ == "__main__":
    main() 