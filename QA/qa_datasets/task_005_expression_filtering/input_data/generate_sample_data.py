#!/usr/bin/env python3
"""
Generate sample expression data for task_005_expression_filtering.
This script creates a small sample from the original TCGA-LUAD data.
"""

import pandas as pd
import numpy as np

# Original data path
original_data_path = '/workdir_efs/jhjeon/Biomni/chainlit/chainlit_logs/38c472b2-c3f9-49c9-af2b-35f5566c60ee/TCGA-LUAD.star_counts.tsv.gz'

# Load original data
print("Loading original data...")
df = pd.read_csv(original_data_path, compression='gzip', sep='\t', index_col=0)

# Select a subset: 500 genes, 50 samples (40 tumor + 10 normal)
sample_cols = df.columns.tolist()
tumor_samples = [col for col in sample_cols if col.split('-')[3][:2] in ['01', '02', '03', '04', '05', '06', '07', '08', '09']]
normal_samples = [col for col in sample_cols if col.split('-')[3][:2] in ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19']]

# Select samples
selected_tumor = tumor_samples[:40]
selected_normal = normal_samples[:10]
selected_samples = selected_tumor + selected_normal

# Select genes (first 500)
selected_genes = df.index[:500].tolist()

# Create subset
sample_df = df.loc[selected_genes, selected_samples]

# Save to TSV
output_path = 'expression_matrix.tsv'
sample_df.to_csv(output_path, sep='\t')
print(f"Sample data saved to {output_path}")
print(f"Shape: {sample_df.shape}")
print(f"Tumor samples: {len(selected_tumor)}, Normal samples: {len(selected_normal)}")
