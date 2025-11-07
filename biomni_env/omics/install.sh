env_name="biomni_hits_omics"

conda env create -n $env_name -f bio_env.yml
conda activate $env_name
pip install -r requirements_langchain.txt
pip install -r requirements.txt

conda env update --file r_packages.yml
Rscript install_r_packages.R

pip install -e ../../../
