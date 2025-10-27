env_name="biomni_hits2"

conda create -n $env_name -f bio_env.yml
conda activate $env_name
pip install -r requirements.txt
pip install -r requirements_langchain.txt
conda env update --file r_packages.yml
Rscript install_r_packages_hits.R

#install plink 2.0
./install_cli_tools.sh --tool "Plink 2.0"

pip install -e ../../
