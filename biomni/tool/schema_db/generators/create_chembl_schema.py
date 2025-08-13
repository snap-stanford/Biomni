import pickle

# ChEMBL schema based on official documentation at https://www.ebi.ac.uk/chembl/api/data/docs
chembl_schema = {
    'base_url': 'https://www.ebi.ac.uk/chembl/api/data',
    'description': 'ChEMBL REST API for bioactivity data',
    'docs_url': 'https://www.ebi.ac.uk/chembl/api/data/docs',
    'license': 'CC BY-SA 3.0',
    
    'resources': {
        'activity': {
            'description': 'Activity values recorded in an Assay',
            'url': '/activity',
            'filters': [
                'assay_chembl_id', 'assay_type', 'assay_variant_mutation', 'assay_variant_sequence',
                'bao_endpoint', 'bao_format', 'bao_label', 'cell_chembl_id', 'confidence_score',
                'data_validity_comment', 'document_chembl_id', 'molecule_chembl_id', 'potential_duplicate',
                'pchembl_value', 'pchembl_value__gte', 'pchembl_value__lte', 'pchembl_value__isnull',
                'published_type', 'published_units', 'published_value', 'published_value__gte',
                'published_value__lte', 'published_value__isnull', 'qudt_units', 'record_id',
                'relation', 'src_id', 'standard_flag', 'standard_relation', 'standard_text_value',
                'standard_type', 'standard_units', 'standard_value', 'standard_value__gte',
                'standard_value__lte', 'standard_value__isnull', 'target_chembl_id', 'target_organism',
                'target_pref_name', 'target_tax_id', 'text_value', 'toid', 'type', 'uo_units',
                'value__gte', 'value__lte', 'value__isnull'
            ]
        },
        'assay': {
            'description': 'Assay details as reported in source Document/Dataset',
            'url': '/assay',
            'filters': [
                'assay_category', 'assay_cell_type', 'assay_chembl_id', 'assay_organism',
                'assay_organism__in', 'assay_strain', 'assay_subcellular_fraction', 'assay_test_type',
                'assay_tissue', 'assay_type', 'assay_variant_accession', 'assay_variant_mutation',
                'bao_format', 'cell_chembl_id', 'confidence_score', 'curated_by', 'description',
                'description__icontains', 'document_chembl_id', 'relationship_description',
                'relationship_type', 'src_assay_id', 'src_id', 'tissue_chembl_id', 'variant_id'
            ]
        },
        'atc_class': {
            'description': 'WHO ATC Classification for drugs',
            'url': '/atc_class',
            'filters': [
                'level1', 'level1_description', 'level2', 'level2_description', 'level3',
                'level3_description', 'level4', 'level4_description', 'level5', 'level5_description',
                'who_name', 'who_name__icontains'
            ]
        },
        'binding_site': {
            'description': 'Target binding site definition',
            'url': '/binding_site',
            'filters': [
                'binding_site_chembl_id', 'site_name', 'site_name__icontains', 'site_type'
            ]
        },
        'biotherapeutic': {
            'description': 'Biotherapeutic molecules',
            'url': '/biotherapeutic',
            'filters': [
                'biotherapeutic_chembl_id', 'description', 'description__icontains', 'helm_notation',
                'helm_notation__icontains', 'molecule_chembl_id', 'molecule_type', 'sequence',
                'sequence__icontains', 'tax_id', 'taxonomy'
            ]
        },
        'cell_line': {
            'description': 'Cell line information',
            'url': '/cell_line',
            'filters': [
                'cell_chembl_id', 'cell_name', 'cell_name__icontains', 'cell_source_organism',
                'cell_source_tissue', 'cell_source_tissue__icontains', 'clo_id', 'efo_id',
                'cellosaurus_id', 'cellosaurus_accession_id'
            ]
        },
        'chembl_id_lookup': {
            'description': 'Look up ChEMBL Id entity type',
            'url': '/chembl_id_lookup',
            'filters': [
                'chembl_id', 'entity_type'
            ]
        },
        'document': {
            'description': 'Document/Dataset from which Assays have been extracted',
            'url': '/document',
            'filters': [
                'doc_id', 'document_chembl_id', 'doi', 'doi__icontains', 'journal', 'journal__icontains',
                'pubmed_id', 'title', 'title__icontains', 'year', 'year__gte', 'year__lte'
            ]
        },
        'mechanism': {
            'description': 'Mechanism of action information for FDA-approved drugs',
            'url': '/mechanism',
            'filters': [
                'action_type', 'direct_interaction', 'disease_efficacy', 'mechanism_comment',
                'mechanism_comment__icontains', 'mechanism_of_action', 'mechanism_of_action__icontains',
                'mechanism_refs', 'molecule_chembl_id', 'record_id', 'selectivity_comment',
                'selectivity_comment__icontains', 'site_id', 'target_chembl_id'
            ]
        },
        'molecule': {
            'description': 'Molecule/biotherapeutics information',
            'url': '/molecule',
            'filters': [
                'availability_type', 'biotherapeutic', 'black_box_warning', 'chebi_par_id',
                'dosed_ingredient', 'first_approval', 'first_approval__gte', 'first_approval__lte',
                'helm_notation', 'helm_notation__icontains', 'indication_class', 'inorganic_flag',
                'max_phase', 'max_phase__gte', 'max_phase__lte', 'molecule_chembl_id',
                'molecule_properties__acd_most_apka', 'molecule_properties__acd_most_bpka',
                'molecule_properties__acd_logp', 'molecule_properties__acd_logp__gte',
                'molecule_properties__acd_logp__lte', 'molecule_properties__alogp',
                'molecule_properties__alogp__gte', 'molecule_properties__alogp__lte',
                'molecule_properties__aromatic_rings', 'molecule_properties__full_molformula',
                'molecule_properties__full_mwt', 'molecule_properties__full_mwt__gte',
                'molecule_properties__full_mwt__lte', 'molecule_properties__hba',
                'molecule_properties__hba__gte', 'molecule_properties__hba__lte',
                'molecule_properties__hbd', 'molecule_properties__hbd__gte',
                'molecule_properties__hbd__lte', 'molecule_properties__molecular_species',
                'molecule_properties__mw_freebase', 'molecule_properties__mw_freebase__gte',
                'molecule_properties__mw_freebase__lte', 'molecule_properties__mw_monoisotopic',
                'molecule_properties__num_ro5_violations', 'molecule_properties__num_ro5_violations__gte',
                'molecule_properties__num_ro5_violations__lte', 'molecule_properties__psa',
                'molecule_properties__psa__gte', 'molecule_properties__psa__lte',
                'molecule_properties__qed_weighted', 'molecule_properties__qed_weighted__gte',
                'molecule_properties__qed_weighted__lte', 'molecule_properties__ro3_pass',
                'molecule_properties__rtb', 'molecule_properties__rtb__gte',
                'molecule_properties__rtb__lte', 'molecule_type', 'natural_product',
                'oral', 'parenteral', 'polymer_flag', 'pref_name', 'pref_name__icontains',
                'prodrug', 'structure_type', 'therapeutic_flag', 'topical', 'usan_stem',
                'usan_stem__icontains', 'usan_substem', 'usan_substem__icontains',
                'usan_year', 'usan_year__gte', 'usan_year__lte', 'withdrawn_class',
                'withdrawn_country', 'withdrawn_flag', 'withdrawn_reason', 'withdrawn_year',
                'withdrawn_year__gte', 'withdrawn_year__lte'
            ]
        },
        'molecule_form': {
            'description': 'Relationships between molecule parents and salts',
            'url': '/molecule_form',
            'filters': [
                'molecule_chembl_id', 'molecule_form_chembl_id', 'parent_chembl_id'
            ]
        },
        'target': {
            'description': 'Targets (protein and non-protein) defined in Assay',
            'url': '/target',
            'filters': [
                'assay_chembl_id', 'binding_site_chembl_id', 'cell_chembl_id', 'confidence_score',
                'cross_references', 'organism', 'organism__in', 'pref_name', 'pref_name__icontains',
                'pref_name__contains', 'protein_classification', 'target_chembl_id',
                'target_components__accession', 'target_components__accession__icontains',
                'target_components__component_id', 'target_components__component_type',
                'target_components__db_source', 'target_components__description',
                'target_components__description__icontains', 'target_components__organism',
                'target_components__organism__in', 'target_components__sequence',
                'target_components__sequence__icontains', 'target_components__tax_id',
                'target_components__target_component_id', 'target_type', 'tax_id'
            ]
        },
        'target_component': {
            'description': 'Target sequence information (A Target may have 1 or more sequences)',
            'url': '/target_component',
            'filters': [
                'accession', 'accession__icontains', 'component_id', 'component_type',
                'db_source', 'description', 'description__icontains', 'organism', 'organism__in',
                'sequence', 'sequence__icontains', 'tax_id', 'target_component_id'
            ]
        },
        'protein_class': {
            'description': 'Protein family classification of TargetComponents',
            'url': '/protein_class',
            'filters': [
                'protein_class_chembl_id', 'protein_class_desc', 'protein_class_desc__icontains',
                'short_name', 'short_name__icontains'
            ]
        },
        'source': {
            'description': 'Document/Dataset source',
            'url': '/source',
            'filters': [
                'src_description', 'src_description__icontains', 'src_id', 'src_short_name',
                'src_short_name__icontains'
            ]
        }
    },
    
    'special_endpoints': {
        'image': {
            'description': 'Graphical (png, svg, json) representation of Molecule',
            'url': '/image/{chembl_id}',
            'formats': ['png', 'svg', 'json'],
            'note': 'Replace {chembl_id} with actual ChEMBL ID'
        },
        'substructure': {
            'description': 'Molecule substructure search',
            'url': '/substructure/{smiles_or_chembl_id}',
            'note': 'Replace {smiles_or_chembl_id} with SMILES string or ChEMBL ID'
        },
        'similarity': {
            'description': 'Molecule similarity search',
            'url': '/similarity/{smiles_or_chembl_id}/{similarity_cutoff}',
            'note': 'Replace {smiles_or_chembl_id} with SMILES string or ChEMBL ID, {similarity_cutoff} with 0-100'
        }
    },
    
    'formats': ['json', 'xml', 'yaml', 'png', 'svg', 'sdf'],
    
    'filter_types': {
        'exact': 'Exact match',
        'iexact': 'Case insensitive exact match',
        'contains': 'Wild card search',
        'icontains': 'Case insensitive wild card search',
        'startswith': 'Starts with query',
        'istartswith': 'Case insensitive starts with',
        'endswith': 'Ends with query',
        'iendswith': 'Case insensitive ends with',
        'gt': 'Greater than',
        'gte': 'Greater than or equal',
        'lt': 'Less than',
        'lte': 'Less than or equal',
        'range': 'Within a range of values',
        'in': 'Appears within list of query values',
        'isnull': 'Field is null'
    },
    
    'common_filters': {
        'max_phase': 'Drug development phase (0-4, where 4=approved)',
        'assay_type': 'Assay type (B=binding, F=functional, A=ADMET, U=unclassified)',
        'standard_type': 'Activity measurement type (IC50, Ki, EC50, etc.)',
        'pchembl_value': 'Negative log of activity value',
        'target_organism': 'Target organism (e.g., Homo sapiens)',
        'molecule_properties__mw_freebase__lte': 'Molecular weight less than or equal to',
        'molecule_properties__alogp__lte': 'LogP less than or equal to',
        'molecule_properties__hba__lte': 'Hydrogen bond acceptors less than or equal to',
        'molecule_properties__hbd__lte': 'Hydrogen bond donors less than or equal to'
    },
    
    'examples': {
        'approved_drugs': 'molecule?max_phase=4',
        'kinase_targets': 'target?pref_name__contains=kinase',
        'molecular_weight_filter': 'molecule?molecule_properties__mw_freebase__lte=300',
        'binding_assays': 'assay?assay_type=B',
        'ic50_activities': 'activity?standard_type=IC50&pchembl_value__gte=6',
        'human_targets': 'target?organism=Homo sapiens',
        'similarity_search': 'similarity/CC(=O)Oc1ccccc1C(=O)O/80',
        'substructure_search': 'substructure/CN(CCCN)c1cccc2ccccc12',
        'molecule_image': 'image/CHEMBL25.svg',
        'mechanism_of_action': 'mechanism?molecule_chembl_id=CHEMBL25',
        'protein_classification': 'protein_class?protein_class_desc__icontains=kinase'
    }
}

with open('chembl.pkl', 'wb') as f:
    pickle.dump(chembl_schema, f)

print('ChEMBL schema created successfully')
