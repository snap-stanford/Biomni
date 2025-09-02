# Known Package Conflicts in Biomni

This file lists Python packages that are known to have dependency conflicts with the default Biomni environment. These packages are not installed by default. If you require their functionality, you must install them manually and may need to uncomment relevant code in the codebase.

## Packages

### 1. hyperimpute
- Not installed by default due to dependency conflicts with the main environment.
- If you need imputation tools that require this package, install it manually in a separate environment or with caution.

### 2. langchain_aws
- Needed for Amazon Bedrock support.
- Amazon Bedrock support is present in the codebase, but due to package dependency conflicts, you should install `langchain_aws` only when you need Bedrock support.
- You must also uncomment the relevant Bedrock support code sections in the codebase to enable this feature.

### 3. cnvkit
- **Environment Requirement**: Requires Python 3.10 environment (`bio_env_py310.yml`)
- **Function**: Supports the `analyze_copy_number_purity_ploidy_and_focal_events` function
- **Why Separate Environment**: cnvkit has strict dependency requirements that conflict with newer Python versions and other packages in the main Biomni environment. Python 3.10 provides the optimal compatibility for cnvkit and its dependencies, ensuring reliable copy number variation analysis, purity estimation, ploidy detection, and focal event identification.
- **Installation**: Use `conda env create -f bio_env_py310.yml` to create the dedicated environment

### 4. panhumanpy
- **Environment Requirement**: Requires its own dedicated environment due to specific dependency constraints
- **Function**: Supports the `annotate_celltype_with_panhumanpy` function for automated cell type annotation in scRNA-seq data
- **Why Separate Environment**: panhumanpy has strict version requirements including:
  - TensorFlow 2.17 (specific version)
  - scikit-learn 1.6.0 (specific version)
  - Python >=3.9 but optimized for Python 3.12
  - These version constraints conflict with the main Biomni environment and other packages
- **Installation**: Install in a separate conda environment with the exact versions specified in the package requirements

---

If you encounter other package conflicts, please add them to this file or open an issue.
