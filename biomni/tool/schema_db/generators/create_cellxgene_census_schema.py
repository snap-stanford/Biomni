import pickle

# CELLxGENE Census Python API schema
cellxgene_census_schema = {
    'type': 'python_api',
    'description': 'CELLxGENE Census Python API for querying single-cell data corpus',
    'docs_url': 'https://chanzuckerberg.github.io/cellxgene-census/python-api.html',
    'github_url': 'https://github.com/chanzuckerberg/cellxgene-census',
    'license': 'MIT license',
    'data_license': 'CC-BY license',
    'package_name': 'cellxgene-census',
    'import_statement': 'import cellxgene_census',
    
    'main_functions': {
        'open_soma': {
            'description': 'Open the Census by version or URI',
            'function': 'cellxgene_census.open_soma',
            'parameters': {
                'census_version': 'Version of Census to open (default: latest)',
                'uri': 'Direct URI to Census data',
                'context': 'SOMA context for configuration'
            }
        },
        'get_anndata': {
            'description': 'Get a slice of Census data as AnnData object',
            'function': 'cellxgene_census.get_anndata',
            'parameters': {
                'census': 'Open Census object',
                'organism': 'Organism name (homo_sapiens, mus_musculus)',
                'obs_query': 'Query for observations (cells)',
                'var_query': 'Query for variables (genes)',
                'X_name': 'Expression matrix name (raw, normalized)',
                'obs_column_names': 'Specific obs columns to include',
                'var_column_names': 'Specific var columns to include'
            }
        },
        'get_obs': {
            'description': 'Get observation (cell) metadata',
            'function': 'cellxgene_census.get_obs',
            'parameters': {
                'census': 'Open Census object',
                'organism': 'Organism name',
                'obs_query': 'Query for observations',
                'column_names': 'Specific columns to include'
            }
        },
        'get_var': {
            'description': 'Get variable (gene) metadata',
            'function': 'cellxgene_census.get_var',
            'parameters': {
                'census': 'Open Census object',
                'organism': 'Organism name',
                'var_query': 'Query for variables',
                'column_names': 'Specific columns to include'
            }
        },
        'get_presence_matrix': {
            'description': 'Get feature dataset presence matrix',
            'function': 'cellxgene_census.get_presence_matrix',
            'parameters': {
                'census': 'Open Census object',
                'organism': 'Organism name'
            }
        }
    },
    
    'versioning_functions': {
        'get_census_version_description': {
            'description': 'Get release description for Census version',
            'function': 'cellxgene_census.get_census_version_description'
        },
        'get_census_version_directory': {
            'description': 'Get directory of available Census versions',
            'function': 'cellxgene_census.get_census_version_directory'
        }
    },
    
    'data_download_functions': {
        'get_source_h5ad_uri': {
            'description': 'Get URI for source H5AD dataset',
            'function': 'cellxgene_census.get_source_h5ad_uri'
        },
        'download_source_h5ad': {
            'description': 'Download source H5AD dataset',
            'function': 'cellxgene_census.download_source_h5ad'
        }
    },
    
    'experimental_ml': {
        'pytorch_dataloader': {
            'description': 'Create PyTorch DataLoader for Census data',
            'function': 'cellxgene_census.experimental.ml.pytorch.experiment_dataloader'
        },
        'experiment_datapipe': {
            'description': 'PyTorch DataPipe for Census experiments',
            'function': 'cellxgene_census.experimental.ml.pytorch.ExperimentDataPipe'
        },
        'encoders': {
            'label_encoder': 'cellxgene_census.experimental.ml.encoders.LabelEncoder',
            'batch_encoder': 'cellxgene_census.experimental.ml.encoders.BatchEncoder'
        },
        'huggingface': {
            'cell_dataset_builder': 'cellxgene_census.experimental.ml.huggingface.CellDatasetBuilder',
            'geneformer_tokenizer': 'cellxgene_census.experimental.ml.huggingface.GeneformerTokenizer'
        }
    },
    
    'experimental_processing': {
        'get_highly_variable_genes': {
            'description': 'Get highly variable genes from Census data',
            'function': 'cellxgene_census.experimental.pp.get_highly_variable_genes'
        },
        'highly_variable_genes': {
            'description': 'Identify highly variable genes',
            'function': 'cellxgene_census.experimental.pp.highly_variable_genes'
        },
        'mean_variance': {
            'description': 'Calculate mean and variance',
            'function': 'cellxgene_census.experimental.pp.mean_variance'
        }
    },
    
    'experimental_embeddings': {
        'get_embedding': {
            'description': 'Read cell embeddings as numpy array',
            'function': 'cellxgene_census.experimental.get_embedding'
        },
        'get_embedding_metadata': {
            'description': 'Read embedding metadata',
            'function': 'cellxgene_census.experimental.get_embedding_metadata'
        },
        'get_all_available_embeddings': {
            'description': 'Get all available embeddings for Census version',
            'function': 'cellxgene_census.experimental.get_all_available_embeddings'
        }
    },
    
    'supported_organisms': [
        'homo_sapiens',
        'mus_musculus'
    ],
    
    'expression_matrices': [
        'raw',
        'normalized'
    ],
    
    'common_obs_columns': [
        'cell_type',
        'tissue',
        'disease',
        'sex',
        'ethnicity',
        'development_stage',
        'organism',
        'assay',
        'dataset_id',
        'donor_id',
        'suspension_type',
        'is_primary_data'
    ],
    
    'common_var_columns': [
        'feature_id',
        'feature_name',
        'feature_biotype',
        'nnz',
        'n_measured_obs'
    ],
    
    'query_syntax': {
        'description': 'SOMA query syntax for filtering data',
        'examples': {
            'cell_type_filter': "obs_query=\"cell_type == 'T cell'\"",
            'tissue_filter': "obs_query=\"tissue == 'lung'\"",
            'multiple_conditions': "obs_query=\"cell_type == 'T cell' and tissue == 'lung'\"",
            'gene_filter': "var_query=\"feature_name == 'CD3E'\"",
            'gene_list': "var_query=\"feature_name in ['CD3E', 'CD4', 'CD8A']\""
        }
    },
    
    'installation': {
        'pip': 'pip install cellxgene-census',
        'conda': 'conda install -c conda-forge cellxgene-census'
    },
    
    'basic_usage_example': """
import cellxgene_census

# Open the latest Census
with cellxgene_census.open_soma() as census:
    # Get human T cells from lung tissue
    adata = cellxgene_census.get_anndata(
        census,
        organism="homo_sapiens",
        obs_query="cell_type == 'T cell' and tissue == 'lung'",
        var_query="feature_name in ['CD3E', 'CD4', 'CD8A']"
    )
""",
    
    'data_access_patterns': {
        'full_organism': 'Access all data for an organism',
        'filtered_cells': 'Filter cells by metadata (cell type, tissue, etc.)',
        'filtered_genes': 'Filter genes by name or biotype',
        'specific_datasets': 'Access specific datasets by ID',
        'embeddings': 'Access pre-computed embeddings',
        'presence_matrix': 'Check which genes are measured in which datasets'
    },
    
    'performance_notes': {
        'lazy_loading': 'Data is loaded lazily for memory efficiency',
        'cloud_optimized': 'Optimized for cloud-based access',
        'chunked_access': 'Large queries are automatically chunked',
        'caching': 'Results can be cached for repeated access'
    }
}

with open('cellxgene_census.pkl', 'wb') as f:
    pickle.dump(cellxgene_census_schema, f)

print('CELLxGENE Census schema created successfully')
