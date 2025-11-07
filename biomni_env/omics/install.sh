env_name="hits_omics"

# Install mamba if not already installed
conda install -n base mamba -c conda-forge -y

# Create environment with all dependencies (conda + pip) from unified YAML
mamba env create -v -n $env_name -f bio_env.yml

# Activate environment
conda activate $env_name

# Install R packages if needed
conda env update --file r_packages.yml
Rscript install_r_packages.R

# Install the main package in editable mode
pip install -e ../../../
