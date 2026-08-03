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
# Optional: DeepSpot-M, used by predict_spatial_gene_expression_from_histology.
# The weights are gated: request access at https://huggingface.co/ratschlab/DeepSpotM
# and run `huggingface-cli login`. Non-commercial research use only.
pip install deepspotm || echo "Optional dependency deepspotm was not installed; predict_spatial_gene_expression_from_histology will be unavailable"
