Training datasets
=================

This folder holds the training datasets used in the publication [Lib-INVENT: Reaction Based Generative Scaffold Decoration for in silico Library Design](https://chemrxiv.org/articles/preprint/Lib-INVENT_Reaction_Based_Generative_Scaffold_Decoration_for_in_silico_Library_Design/14473980).

The [datasets](https://doi.org/10.5281/zenodo.6627127) used for training of the prior:
- `purged_chembl_sliced.smi.gz`: The CHEMBl 27 compounds, filtered according to the rules described in the manuscript and sliced according to the reaction rules.
- `chembl_train.smi.gz`: The purged, sliced dataset used for model training. The DRD2 compounds are removed as described in the manuscript.

Run the script `download_training_datasets.py` to download the datasets from [Zenodo](https://zenodo.org/).
