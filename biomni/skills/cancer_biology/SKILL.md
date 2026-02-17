---
name: cancer_biology
description: Analyze cancer-related data, DDR networks, and cancer genomics or transcriptomics.
---

## Tools
- **analyze_ddr_network_in_cancer**: Analyze DNA Damage Response (DDR) network alterations and dependencies in cancer samples.
- **analyze_cell_senescence_and_apoptosis**: Analyze flow cytometry data to quantify senescent and apoptotic cell populations.
- **detect_and_annotate_somatic_mutations**: Detects and annotates somatic mutations in tumor samples compared to matched normal samples using GATK Mutect2 for variant calling, GATK FilterMutectCalls for filtering, and SnpEff for functional annotation.
- **detect_and_characterize_structural_variations**: Detects and characterizes structural variations (SVs) in genomic sequencing data using LUMPY for SV detection followed by annotation with COSMIC and/or ClinVar databases.
- **perform_gene_expression_nmf_analysis**: Performs Non-negative Matrix Factorization (NMF) on gene expression data to extract metagenes and their associated sample weights for tumor subtype identification.
- **analyze_copy_number_purity_ploidy_and_focal_events**: CNVkit-based copy number workflow performing CNV segmentation, purity & ploidy approximation, simplified HRD-style metrics, and focal amplification/deletion detection in selected genes.
