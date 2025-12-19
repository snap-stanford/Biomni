env_name="biomni_hits2"

# Install mamba if not already installed
# conda install -n base mamba -c conda-forge -y
mamba env create -n $env_name -f environment.yml

# Activate environment
source activate $env_name

mamba env update --file bio_env.yml

# Install R packages if needed
mamba env update --file r_packages.yml
Rscript install_r_packages.R

# Install the main package in editable mode
pip install -e ../../
