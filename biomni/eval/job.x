#!/bin/bash
#SBATCH --job-name="biomni_benchmark"
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=100:00:00
#SBATCH -o %x_%a.o%j
#SBATCH -e %x_%a.e%j

cd $SLURM_SUBMIT_DIR
source activate biomni_hits

python benchmark.py -f gemini-3-pro-preview_2 \
                    -l gemini-3-pro-preview \
                    -n 16 \
                    -d gwas_causal_gene_gwas_catalog DbQA gwas_causal_gene_opentargets gwas_variant_prioritization \
                    -s
