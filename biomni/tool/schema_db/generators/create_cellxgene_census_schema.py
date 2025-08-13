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
        'get_default_soma_context': {
            'description': 'Return a tiledbsoma.SOMATileDBContext with sensible defaults',
            'function': 'cellxgene_census.get_default_soma_context',
            'parameters': {}
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
            'description': 'Read the feature dataset presence matrix and return as scipy.sparse.csr_array',
            'function': 'cellxgene_census.get_presence_matrix',
            'parameters': {
                'census': 'Open Census object',
                'organism': 'Organism name'
            }
        }
    },
    
    'versioning_functions': {
        'get_census_version_description': {
            'description': 'Get release description for given Census version, from the Census release directory',
            'function': 'cellxgene_census.get_census_version_description',
            'parameters': {
                'census_version': 'Version of Census to get description for'
            }
        },
        'get_census_version_directory': {
            'description': 'Get the directory of Census versions currently available, optionally filtering by specified flags',
            'function': 'cellxgene_census.get_census_version_directory',
            'parameters': {
                'flags': 'Optional flags to filter versions'
            }
        }
    },
    
    'data_download_functions': {
        'get_source_h5ad_uri': {
            'description': 'Open the named version of the census, and return the URI for the dataset_id',
            'function': 'cellxgene_census.get_source_h5ad_uri',
            'parameters': {
                'census_version': 'Version of Census to use',
                'dataset_id': 'ID of the dataset to get URI for'
            }
        },
        'download_source_h5ad': {
            'description': 'Download the source H5AD dataset, for the given dataset_id, to the user-specified file name',
            'function': 'cellxgene_census.download_source_h5ad',
            'parameters': {
                'census_version': 'Version of Census to use',
                'dataset_id': 'ID of the dataset to download',
                'filename': 'File name to save the H5AD file to'
            }
        }
    },
    
    'experimental_ml': {
        'pytorch': {
            'experiment_dataloader': {
                'description': 'Factory method for torch.utils.data.DataLoader',
                'function': 'cellxgene_census.experimental.ml.pytorch.experiment_dataloader'
            },
            'ExperimentDataPipe': {
                'description': 'An torchdata.datapipes.iter.IterDataPipe that reads obs and X data from a tiledbsoma.Experiment',
                'function': 'cellxgene_census.experimental.ml.pytorch.ExperimentDataPipe'
            },
            'Stats': {
                'description': 'Statistics about the data retrieved by ExperimentDataPipe via SOMA API',
                'function': 'cellxgene_census.experimental.ml.pytorch.Stats'
            }
        },
        'encoders': {
            'Encoder': {
                'description': 'Base class for obs encoders',
                'function': 'cellxgene_census.experimental.ml.encoders.Encoder'
            },
            'LabelEncoder': {
                'description': 'Default encoder based on sklearn.preprocessing.LabelEncoder',
                'function': 'cellxgene_census.experimental.ml.encoders.LabelEncoder'
            },
            'BatchEncoder': {
                'description': 'An encoder that concatenates and encodes several obs columns',
                'function': 'cellxgene_census.experimental.ml.encoders.BatchEncoder'
            }
        },
        'huggingface': {
            'CellDatasetBuilder': {
                'description': 'Abstract base class for methods to process CELLxGENE Census ExperimentAxisQuery results into a Hugging Face Dataset',
                'function': 'cellxgene_census.experimental.ml.huggingface.CellDatasetBuilder'
            },
            'GeneformerTokenizer': {
                'description': 'Generate a Hugging Face Dataset containing Geneformer token sequences for each cell in CELLxGENE Census ExperimentAxisQuery results (human)',
                'function': 'cellxgene_census.experimental.ml.huggingface.GeneformerTokenizer'
            }
        }
    },
    
    'experimental_processing': {
        'get_highly_variable_genes': {
            'description': 'Convenience wrapper around tiledbsoma.Experiment query and highly_variable_genes function',
            'function': 'cellxgene_census.experimental.pp.get_highly_variable_genes'
        },
        'highly_variable_genes': {
            'description': 'Identify and annotate highly variable genes contained in the query results',
            'function': 'cellxgene_census.experimental.pp.highly_variable_genes'
        },
        'mean_variance': {
            'description': 'Calculate mean and/or variance along the obs axis from query results',
            'function': 'cellxgene_census.experimental.pp.mean_variance'
        }
    },
    
    'experimental_embeddings': {
        'get_embedding': {
            'description': 'Read cell (obs) embeddings and return as a dense numpy.ndarray',
            'function': 'cellxgene_census.experimental.get_embedding',
            'parameters': {
                'census': 'Open Census object',
                'organism': 'Organism name',
                'embedding_name': 'Name of the embedding to retrieve',
                'obs_query': 'Query for observations'
            }
        },
        'get_embedding_metadata': {
            'description': 'Read embedding metadata and return as a Python dict',
            'function': 'cellxgene_census.experimental.get_embedding_metadata',
            'parameters': {
                'census': 'Open Census object',
                'organism': 'Organism name'
            }
        },
        'get_embedding_metadata_by_name': {
            'description': 'Return metadata for a specific embedding',
            'function': 'cellxgene_census.experimental.get_embedding_metadata_by_name',
            'parameters': {
                'census': 'Open Census object',
                'organism': 'Organism name',
                'embedding_name': 'Name of the embedding'
            }
        },
        'get_all_available_embeddings': {
            'description': 'Return a dictionary of all available embeddings for a given Census version',
            'function': 'cellxgene_census.experimental.get_all_available_embeddings',
            'parameters': {
                'census_version': 'Version of Census to check'
            }
        },
        'get_all_census_versions_with_embedding': {
            'description': 'Get a list of all census versions that contain a specific embedding',
            'function': 'cellxgene_census.experimental.get_all_census_versions_with_embedding',
            'parameters': {
                'embedding_name': 'Name of the embedding to search for'
            }
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
            'gene_list': "var_query=\"feature_name in ['CD3E', 'CD4', 'CD8A']\"",
            'in_operator': "obs_query=\"cell_type in ['T cell', 'B cell', 'NK cell']\"",
            'not_operator': "obs_query=\"disease != 'normal'\"",
            'numeric_comparison': "obs_query=\"n_measured_vars > 1000\""
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
    
    # Get cell metadata only
    obs_df = cellxgene_census.get_obs(
        census,
        organism="homo_sapiens",
        obs_query="tissue == 'brain'"
    )
    
    # Get gene metadata only
    var_df = cellxgene_census.get_var(
        census,
        organism="homo_sapiens",
        var_query="feature_biotype == 'protein_coding'"
    )
""",
    
    'data_access_patterns': {
        'full_organism': 'Access all data for an organism',
        'filtered_cells': 'Filter cells by metadata (cell type, tissue, etc.)',
        'filtered_genes': 'Filter genes by name or biotype',
        'specific_datasets': 'Access specific datasets by ID',
        'embeddings': 'Access pre-computed embeddings',
        'presence_matrix': 'Check which genes are measured in which datasets',
        'metadata_only': 'Get only cell or gene metadata without expression data'
    },
    
    'performance_notes': {
        'lazy_loading': 'Data is loaded lazily for memory efficiency',
        'cloud_optimized': 'Optimized for cloud-based access',
        'chunked_access': 'Large queries are automatically chunked',
        'caching': 'Results can be cached for repeated access',
        'context_manager': 'Always use context manager (with statement) for proper resource management'
    },
    
    'error_handling': {
        'connection_issues': 'Handle network connectivity issues gracefully',
        'memory_management': 'Large queries may require significant memory',
        'timeout_handling': 'Long-running queries may timeout',
        'version_compatibility': 'Ensure Census version compatibility'
    }
}

with open('cellxgene_census.pkl', 'wb') as f:
    pickle.dump(cellxgene_census_schema, f)

print('CELLxGENE Census schema created successfully')
