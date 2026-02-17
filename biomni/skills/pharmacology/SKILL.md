---
name: Pharmacology
description: Drug discovery, ADMET, drug interactions, repurposing, and pharmacology datasets.
---

## Tools
- **run_diffdock_with_smiles**: Run DiffDock molecular docking using a protein PDB file and a SMILES string for the ligand, executing the process in a Docker container.
- **docking_autodock_vina**: Performs molecular docking using AutoDock Vina to predict binding affinities between small molecules and a receptor protein.
- **run_autosite**: Runs AutoSite on a PDB file to identify potential binding sites and returns a research log with the results.
- **retrieve_topk_repurposing_drugs_from_disease_txgnn**: Computes TxGNN model predictions for drug repurposing and returns the top predicted drugs with their scores for a given disease.
- **predict_admet_properties**: Predicts ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties for a list of compounds using pretrained models.
- **predict_binding_affinity_protein_1d_sequence**: Predicts binding affinity between small molecules and a protein sequence using pre-trained deep learning models.
- **analyze_accelerated_stability_of_pharmaceutical_formulations**: Analyzes the stability of pharmaceutical formulations under accelerated storage conditions.
- **run_3d_chondrogenic_aggregate_assay**: Generates a detailed protocol for performing a 3D chondrogenic aggregate culture assay to evaluate compounds' effects on chondrogenesis.
- **grade_adverse_events_using_vcog_ctcae**: Grade and monitor adverse events in animal studies using the VCOG-CTCAE standard.
- **analyze_radiolabeled_antibody_biodistribution**: Analyze biodistribution and pharmacokinetic profile of radiolabeled antibodies.
- **estimate_alpha_particle_radiotherapy_dosimetry**: Estimate radiation absorbed doses to tumor and normal organs for alpha-particle radiotherapeutics using the Medical Internal Radiation Dose (MIRD) schema.
- **perform_mwas_cyp2c19_metabolizer_status**: Perform a Methylome-wide Association Study (MWAS) to identify CpG sites significantly associated with CYP2C19 metabolizer status.
- **calculate_physicochemical_properties**: Calculate key physicochemical properties of a drug candidate molecule.
- **analyze_xenograft_tumor_growth_inhibition**: Analyze tumor growth inhibition in xenograft models across different treatment groups.
- **analyze_western_blot**: Performs densitometric analysis of Western blot images to quantify relative protein expression.
- **query_drug_interactions**: Query drug-drug interactions from DDInter database to identify potential interactions, mechanisms, and severity levels between specified drugs.
- **check_drug_combination_safety**: Analyze safety of a drug combination for potential interactions using DDInter database with comprehensive risk assessment and clinical recommendations.
- **analyze_interaction_mechanisms**: Analyze interaction mechanisms between two specific drugs providing detailed mechanistic insights and clinical significance assessment.
- **find_alternative_drugs_ddinter**: Find alternative drugs that don't interact with contraindicated drugs using DDInter database for safer therapeutic substitutions.
- **query_fda_adverse_events**: Query FDA adverse event reports for specific drugs from the OpenFDA database to identify potential safety signals, reaction patterns, and regulatory intelligence.
- **get_fda_drug_label_info**: Retrieve FDA drug label information including indications, contraindications, warnings, and dosage information from the OpenFDA database.
- **check_fda_drug_recalls**: Check for FDA drug recalls and enforcement actions from the OpenFDA database to identify safety concerns and regulatory actions.
- **analyze_fda_safety_signals**: Analyze safety signals across multiple drugs using OpenFDA adverse event data to identify patterns and comparative risk profiles.
