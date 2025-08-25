| **Component Requiring Commercial License** | **Type**         | **Original Location in Codebase**                                    | **Permissive Alternative**                               |
|-------------------------------------------|------------------|-----------------------------------------------------------------------|-----------------------------------------------------------|
| KEGG API                                  | API / Database   | `biomni/tool/database.py: query_kegg`                                | Reactome API (CC-BY 4.0)                                 |
| HOMER                                     | CLI Tool         | `biomni/tool/genomics.py: find_enriched_motifs_with_homer`           | BioPython motifs + JASPAR API (Permissive)               |

| OMIM                                      | Data Source      | `biomni/env_desc.py: data_lake_dict (omim.parquet)`                  | ClinVar API and Monarch Initiative API (Public Domain)   |
| DDInter 2.0                               | Data Source      | `biomni/env_desc.py: data_lake_dict (all ddinter_... files)`         | OpenFDA Adverse Event API (Public Domain)                |
| Human Protein Atlas                       | Data Source      | `biomni/env_desc.py: data_lake_dict (proteinatlas.tsv)`              | Ensembl API or UniProt API (Permissive)                  |
| Guide to PHARMACOLOGY (GtoPdb)            | API / Database   | `biomni/tool/database.py: query_gtopdb`                              | ChEMBL API (Permissive)                                  |
