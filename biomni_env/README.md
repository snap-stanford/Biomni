# Biomni Environment Setup

This directory contains scripts and configuration files to set up a comprehensive bioinformatics environment with various tools and packages.

1. Clone the repository:
   ```bash
   git clone https://github.com/snap-stanford/Biomni.git
   cd Biomni/biomni_env
   ```

2. Setting up the environment:
- (a) If you want to use or try out the basic agent without the full E1 or install your own softwares, run the following script:

```bash
conda env create -f environment.yml
```

- (b) If you want to use the full environment E1, run the setup script (this script takes > 10 hours to setup, and requires a disk of at least 30 GB quota). Follow the prompts to install the desired components.

```bash
bash setup.sh
```

If you already installed the base version, and just wants to add the additional packages in the new release, you can simply do:

```bash
bash new_software_v005.sh
```

Note: we have only tested this setup.sh script with Ubuntu 22.04, 64 bit.

- (c) If you want to use a reduced conda environment without R or CLI tools, run the following script:

```bash
conda env create -f fixed_env.yml
```

This contains most of the packages from environment.yml and bio_env.yml, and requires a disk of at elast 13GB quota.

- (d) **Python 3.10 Environment for Copy Number Analysis**: If you specifically need to use the `analyze_copy_number_purity_ploidy_and_focal_events` function, we provide a Python 3.10 environment option. This function has specific dependency requirements that are best met with Python 3.10. To set up this environment:

```bash
conda env create -f bio_env_py310.yml
```

This environment is optimized for copy number variation analysis and includes the necessary packages for purity, ploidy, and focal event detection.

3. Lastly, to activate the biomni environment:
```bash
conda activate biomni_e1
```

For the Python 3.10 environment specifically:
```bash
conda activate biomni_py310
```

### 📦 Langchain Package Support

The Biomni environment comes with a minimal set of langchain packages by default:
- `langchain-openai` - for OpenAI model support
- `langchain-anthropic` - for Anthropic model support
- `langchain-ollama` - for Ollama model support

If you need support for other external models or services, you'll need to install additional langchain packages manually. For example:

```bash
# For AWS Bedrock support
pip install langchain-aws

# For Google Gemini support
pip install langchain-google-genai

```
