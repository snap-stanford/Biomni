
# =============================================================================
# CONFIGURATION PARAMETERS - Modify these as needed
# =============================================================================

import os

# Base paths
BIOMNI_ROOT = "/home/igor/exploration/biomni/Biomni/"
TUTORIALS_DIR = os.path.join(BIOMNI_ROOT, "tutorials")
DATA_DIR = os.path.join(TUTORIALS_DIR, "data")
EMBEDDINGS_DIR = os.path.join(TUTORIALS_DIR, "zero-shot-performance", "embeddings")

# Dataset parameters
SYNTHETIC_FILENAME = "synthetic_ensembl_dataset.h5ad"
SYNTHETIC_EMBEDDINGS_PREFIX = "synthetic_geneformer_embeddings"
SYNTHETIC_UMAP_PLOT_FILENAME = "synthetic_geneformer_umap.png"

# Geneformer parameters
# Available models:
# - "Geneformer-V1-10M": 10M parameters, fastest, good for quick testing
# - "Geneformer-V2-104M": 104M parameters, balanced performance (default)
# - "Geneformer-V2-104M_CLcancer": 104M parameters, cancer-specific fine-tuning
# - "Geneformer-V2-316M": 316M parameters, highest performance, requires more memory
MODEL_NAME = "Geneformer-V2-104M"
MODEL_INPUT_SIZE = 4096
CHUNK_SIZE = 10000
NPROC = 8
FORWARD_BATCH_SIZE = 64

# Synthetic dataset parameters
N_CELLS = 1000
N_GENES = 2000
N_CELL_TYPES = 5

# UMAP parameters
UMAP_SIZE = 5
UMAP_DPI = 300

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_synthetic_ensembl_dataset(n_cells, n_genes, n_cell_types, random_seed=42):
    """
    Create a synthetic single-cell RNA-seq dataset with Ensembl gene IDs.

    Parameters
    ----------
    n_cells : int
        Number of cells to generate
    n_genes : int
        Number of genes to generate
    n_cell_types : int
        Number of different cell types
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    adata : AnnData
        Synthetic AnnData object with Ensembl gene IDs
    """
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy import sparse

    np.random.seed(random_seed)

    # Generate synthetic Ensembl gene IDs
    # Real Ensembl IDs follow pattern: ENSG + 11 digits
    ensembl_ids = []
    for i in range(n_genes):
        # Generate 11-digit number with leading zeros
        gene_number = f"{i+1:011d}"
        ensembl_id = f"ENSG{gene_number}"
        ensembl_ids.append(ensembl_id)

    # Generate synthetic cell type labels
    cell_types = [f"CellType_{i+1}" for i in range(n_cell_types)]
    cell_type_labels = np.random.choice(cell_types, size=n_cells)

    # Generate synthetic expression matrix
    # Use negative binomial distribution to simulate count data
    expression_matrix = np.zeros((n_cells, n_genes))

    for i, cell_type in enumerate(cell_types):
        # Get cells of this type
        cell_mask = cell_type_labels == cell_type

        # Generate different expression patterns for each cell type
        # Some genes are highly expressed in specific cell types
        for j in range(n_genes):
            if j % (n_genes // n_cell_types) == i:
                # High expression for cell type-specific genes
                mean_expr = np.random.negative_binomial(5, 0.3, size=np.sum(cell_mask))
            else:
                # Low baseline expression
                mean_expr = np.random.negative_binomial(2, 0.7, size=np.sum(cell_mask))

            expression_matrix[cell_mask, j] = mean_expr

    # Add some noise
    noise = np.random.poisson(1, size=(n_cells, n_genes))
    expression_matrix = expression_matrix + noise

    # Convert to sparse matrix for efficiency
    expression_matrix = sparse.csr_matrix(expression_matrix.astype(int))

    # Create cell metadata
    cell_metadata = pd.DataFrame({
        'cell_type': cell_type_labels,
        'cell_id': [f"Cell_{i:04d}" for i in range(n_cells)],
        'n_counts': np.array(expression_matrix.sum(axis=1)).flatten(),
        'n_genes': np.array((expression_matrix > 0).sum(axis=1)).flatten()
    })

    # Create gene metadata
    gene_metadata = pd.DataFrame({
        'gene_id': ensembl_ids,
        'gene_symbol': [f"GENE_{i+1:04d}" for i in range(n_genes)],
        'n_cells': np.array((expression_matrix > 0).sum(axis=0)).flatten(),
        'mean_counts': np.array(expression_matrix.mean(axis=0)).flatten()
    })

    # Create AnnData object
    adata = sc.AnnData(
        X=expression_matrix,
        obs=cell_metadata,
        var=gene_metadata
    )

    # Set gene names to Ensembl IDs
    adata.var_names = ensembl_ids
    adata.var_names_unique = ensembl_ids

    # Add some basic preprocessing
    adata.var['highly_variable'] = adata.var['n_cells'] > n_cells * 0.1
    adata.obs['total_counts'] = adata.obs['n_counts']

    return adata

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import torch
    import scanpy as sc
    import numpy as np
    import pandas as pd

    import sys
    sys.path.append(BIOMNI_ROOT)
    from biomni.tool.genomics import geneformer_embed

    print("=" * 80)
    print("GENERATING SYNTHETIC DATASET WITH ENSEMBL IDs")
    print("=" * 80)

    # Create synthetic dataset with Ensembl IDs
    print(f"Creating synthetic dataset with {N_CELLS} cells, {N_GENES} genes, {N_CELL_TYPES} cell types...")
    synthetic_adata = create_synthetic_ensembl_dataset(N_CELLS, N_GENES, N_CELL_TYPES)

    print(f"Synthetic dataset created: {synthetic_adata}")
    print(f"Number of cells: {synthetic_adata.n_obs}")
    print(f"Number of genes: {synthetic_adata.n_vars}")
    print(f"Sample Ensembl IDs: {list(synthetic_adata.var_names[:10])}")
    print(f"Cell types: {list(synthetic_adata.obs['cell_type'].unique())}")

    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # Save synthetic dataset
    synthetic_path = os.path.join(DATA_DIR, SYNTHETIC_FILENAME)
    synthetic_adata.write(synthetic_path)
    print(f"Saved synthetic dataset to: {synthetic_path}")

    print("\n" + "=" * 80)
    print("TESTING WITH SYNTHETIC DATASET (ENSEMBL IDs)")
    print("=" * 80)

    # Test with synthetic dataset (should work better with Ensembl IDs)
    print("Testing Geneformer with synthetic dataset containing Ensembl IDs...")
    synthetic_steps = geneformer_embed(
        adata_or_path=synthetic_adata,
        base_dir=TUTORIALS_DIR,
        adata_filename=SYNTHETIC_FILENAME,
        embeddings_prefix=SYNTHETIC_EMBEDDINGS_PREFIX,
        model_name=MODEL_NAME,
        model_input_size=MODEL_INPUT_SIZE,
        chunk_size=CHUNK_SIZE,
        nproc=NPROC,
        forward_batch_size=FORWARD_BATCH_SIZE,
    )
    print("Synthetic dataset Geneformer embedding steps:")
    print(synthetic_steps)

    # Load the synthetic embeddings CSV
    synthetic_embeddings_path = os.path.join(EMBEDDINGS_DIR, f"{SYNTHETIC_EMBEDDINGS_PREFIX}.csv")
    synthetic_embeddings_df = pd.read_csv(synthetic_embeddings_path, index_col=0)
    print(f"✓ Loaded synthetic embeddings with shape: {synthetic_embeddings_df.shape}")
    print(f"✓ Synthetic embeddings columns: {list(synthetic_embeddings_df.columns)}")

    synthetic_adata.obsm['X_geneformer'] = synthetic_embeddings_df.values
    print("\nChecking for synthetic Geneformer embeddings...")
    print(f"Available obsm keys: {list(synthetic_adata.obsm.keys())}")

    if 'X_geneformer' in synthetic_adata.obsm:
        synthetic_adata.obsm['X_umap_input'] = synthetic_adata.obsm['X_geneformer']
        use_rep = 'X_umap_input'
        print(f"✓ Found synthetic Geneformer embeddings in obsm['X_geneformer'] with shape: {synthetic_adata.obsm['X_geneformer'].shape}")
    else:
        print("⚠️  No synthetic Geneformer embeddings found in expected location (obsm['X_geneformer'])")
        print("⚠️  Using raw data for UMAP.")
        use_rep = None

    sc.pp.neighbors(synthetic_adata, use_rep=use_rep)
    sc.tl.umap(synthetic_adata)
    import matplotlib.pyplot as plt
    synthetic_umap_output_path = os.path.join(DATA_DIR, SYNTHETIC_UMAP_PLOT_FILENAME)
    sc.pl.umap(synthetic_adata, color='cell_type', show=False, size=UMAP_SIZE, title="UMAP of Synthetic Geneformer embeddings")
    plt.savefig(synthetic_umap_output_path, dpi=UMAP_DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Synthetic UMAP plot saved to: {synthetic_umap_output_path}")

    # Show cell type distribution
    print(f"\nCell type distribution in synthetic dataset:")
    print(synthetic_adata.obs['cell_type'].value_counts())

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print("Generated files:")
    print(f"- Synthetic dataset: {synthetic_path}")
    print(f"- Synthetic embeddings: {os.path.join(EMBEDDINGS_DIR, f'{SYNTHETIC_EMBEDDINGS_PREFIX}.csv')}")
    print(f"- Synthetic UMAP plot: {os.path.join(DATA_DIR, SYNTHETIC_UMAP_PLOT_FILENAME)}")
