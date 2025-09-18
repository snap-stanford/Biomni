# Biomni Data Source License Information

This document provides an overview of the data sources used by Biomni, their licenses, and suitability for internal hosting and commercial use.

A significant portion of the data used in Biomni requires a commercial license for any commercial application. Several datasets are explicitly licensed for non-commercial use only, which would prohibit their use in a commercial product without a separate agreement. Before proceeding with any commercial use, a thorough legal review of the licenses for each dataset you intend to use is strongly recommended.

## Data License and Internal Hosting Analysis

| Data Source Category | Example Files | License | Internal Hosting | Source |
| :--- | :--- | :--- | :--- | :--- |
| **COSMIC** | `Cosmic_*.csv`, `Cosmic_*.parquet` | Requires commercial license for commercial use. | Yes, with a valid commercial license. | [Sanger Institute](https://cancer.sanger.ac.uk/cosmic) |
| **BindingDB** | `BindingDB_All_202409.tsv` | Custom, non-commercial use granted. Commercial use requires a license. | Yes, with a commercial license. | [BindingDB](https://www.bindingdb.org) |
| **Broad Repurposing Hub** | `broad_repurposing_hub_*.parquet` | CC BY 4.0 | Yes | [Broad Institute](https://www.broadinstitute.org/drug-repurposing-hub) |
| **DDInter** | `ddinter_*.csv` | CC BY-NC-SA 4.0 | No, non-commercial use only. | [DDInter](http://ddinter.scbdd.com/) |
| **DisGeNET** | `DisGeNET.parquet` | CC BY-NC-SA 4.0 | No, non-commercial use only. | [DisGeNET](https://www.disgenet.org/) |
| **Enamine** | `enamine_cloud_library_smiles.pkl` | Proprietary. Requires license for screening. | Yes, with a valid license. | [Enamine](https://enamine.net/) |
| **EveBio** | `evebio_*.csv` | Appears to be proprietary data from EveBio. | Requires permission from EveBio. | EveBio |
| **Gene Ontology (GO)** | `go-plus.json` | CC BY 4.0 | Yes | [Gene Ontology Consortium](http://geneontology.org/) |
| **GTEx** | `gtex_tissue_gene_tpm.parquet` | dbGaP controlled access. | Yes, with authorization. | [GTEx Portal](https://gtexportal.org/) |
| **Human Protein Atlas** | `proteinatlas.tsv` | CC BY-SA 3.0 | Yes | [Human Protein Atlas](https://www.proteinatlas.org/) |
| **MSigDB** | `msigdb_human_*.parquet` | Custom, requires license for commercial use. | Yes, with a license. | [Broad Institute](https://www.gsea-msigdb.org/gsea/msigdb) |
| **OMIM** | `omim.parquet` | Custom, requires license for commercial use. | Yes, with a license. | [OMIM](https://omim.org/) |
| **BioGRID** | `affinity_capture-ms.parquet`, etc. | OSL 3.0 | Yes | [BioGRID](https://thebiogrid.org/) |
| **CZI Cell Census** | `czi_census_datasets_v4.parquet` | CC BY 4.0 | Yes | [Chan Zuckerberg Initiative](https://cellxgene.cziscience.com/census) |
| **DepMap** | `DepMap_*.csv` | CC BY 4.0 | Yes | [Broad Institute DepMap](https://depmap.org/) |
| **Genebass** | `genebass_*.pkl` | ODC-By v1.0 | Yes | [Genebass](https://genebass.org/) |
| **GWAS Catalog** | `gwas_catalog.pkl` | Apache 2.0 | Yes | [EBI GWAS Catalog](https://www.ebi.ac.uk/gwas/) |
| **HPO** | `hp.obo` | Custom, free for all uses. | Yes | [Human Phenotype Ontology](https://hpo.jax.org/) |
| **McPAS-TCR** | `McPAS-TCR.parquet` | CC BY-NC-SA 4.0 | No, non-commercial use only. | [McPAS-TCR](http://friedmanlab.weizmann.ac.il/McPAS-TCR/) |
| **miRDB** | `miRDB_v6.0_results.parquet` | Custom, free for non-commercial use. | No, non-commercial use only. | [miRDB](http://mirdb.org/) |
| **miRTarBase** | `miRTarBase_*.parquet` | CC BY-NC 4.0 | No, non-commercial use only. | [miRTarBase](https://mirtarbase.cuhk.edu.cn/) |
| **MouseMine** | `mousemine_*.parquet` | CC BY 4.0 | Yes | [MouseMine](http://www.mousemine.org/) |
| **P-HIPSTER** | `Virus-Host_PPI_P-HIPSTER_2020.parquet` | CC BY 4.0 | Yes | [P-HIPSTER](http://phipster.org) |
| **TXGNN** | `txgnn_*.pkl` | MIT License | Yes | - |

## Configuration for Commercial Use

To manage which datasets are used based on licensing, Biomni provides a configuration option. You can set the `commercial_mode` flag to `True` in your configuration to automatically exclude datasets that are not licensed for commercial use.

### Usage

```python
from biomni.agent import A1

# For commercial use (excludes non-commercial datasets)
agent = A1(commercial_mode=True)

# For academic/research use (includes all datasets)
agent = A1(commercial_mode=False)  # default
```

This configuration automatically selects the appropriate data environment description file and ensures compliance with licensing requirements.
