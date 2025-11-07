env_name="hits_omics"
conda install -n base mamba -c conda-forge
conda env create -v -n $env_name -f bio_env.yml
conda activate $env_name
pip install -r requirements_langchain.txt
pip install -r requirements.txt

conda env update --file r_packages.yml
Rscript install_r_packages.R

pip install -e ../../../
