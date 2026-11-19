# Biomni v0.0.8 - Incremental Software Installation
# Add any new packages/software introduced in version 0.0.8 below
pip install transformers sentencepiece langchain-google-genai langchain_ollama mcp
pip install lazyslide
pip install "git+https://github.com/YosefLab/popV.git@refs/pull/100/head"
pip install uv
sudo apt-get install git-lfs # or brew install git-lfs if you are on macOS
git lfs install
pip install pybiomart
pip install fair-esm
pip install uv
uv pip install transcriptformer
pip install "zarr>=2.0,<3.0" #this resolved transcripformer download isses
uv tool install arc-state
pip install nnunet nibabel nilearn
pip install mi-googlesearch-python
pip install git+https://github.com/pylabrobot/pylabrobot.git
conda install weasyprint

# --- Spatial transcriptomics tools (Visium) ---
pip install scanpy squidpy scvi-tools leidenalg scrublet
# NOTE: anndata >=0.13 has a bug writing vlen-string columns with gzip compression
# (file corrupts on read). The spatial tools write h5ad with compression='lzf'
# to avoid this. Do NOT switch those writes back to gzip without testing on
# large (2000+ row) datasets first.
pip install "anndata>=0.10" "scipy>=1.10" "pandas>=2.0" "matplotlib>=3.7"
Rscript biomni_env/install_r_packages_spatial.R
