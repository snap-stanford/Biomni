import pickle

# ChEMBL schema
chembl_schema = {
    'base_url': 'https://www.ebi.ac.uk/chembl/api/data',
    'description': 'ChEMBL REST API for bioactivity data',
    'docs_url': 'https://www.ebi.ac.uk/chembl/api/data/docs',
    'license': 'CC BY-SA 3.0',
    
    'resources': {
        'molecule': {
            'description': 'Molecule information, including properties, structural representations and synonyms',
            'url': '/molecule',
            'filters': ['max_phase', 'molecule_properties__mw_freebase__lte', 'pref_name__iendswith', 'biotherapeutic__isnull']
        },
        'activity': {
            'description': 'Activity values recorded in an Assay',
            'url': '/activity',
            'filters': ['standard_type', 'target_organism__in', 'target_chembl_id__in', 'pchembl_value__gte']
        },
        'assay': {
            'description': 'Assay details as reported in source Document/Dataset',
            'url': '/assay',
            'filters': ['assay_type', 'src_id', 'description__icontains', 'target_chembl_id']
        },
        'target': {
            'description': 'Targets (protein and non-protein) defined in Assay',
            'url': '/target',
            'filters': ['pref_name__contains', 'target_components__accession']
        },
        'mechanism': {
            'description': 'Mechanism of action information for approved drugs',
            'url': '/mechanism',
            'filters': ['molecule_chembl_id', 'mechanism_of_action__icontains', 'action_type']
        },
        'image': {
            'description': 'Graphical (svg) representation of Molecule',
            'url': '/image/{chembl_id}',
            'formats': ['svg']
        },
        'similarity': {
            'description': 'Molecule similarity search',
            'url': '/similarity/{smiles_or_chembl_id}/{similarity_cutoff}'
        },
        'substructure': {
            'description': 'Molecule substructure search',
            'url': '/substructure/{smiles_or_chembl_id}'
        }
    },
    
    'formats': ['json', 'xml', 'yaml', 'svg', 'sdf'],
    
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
    
    'examples': {
        'approved_drugs': 'molecule?max_phase=4',
        'kinase_targets': 'target?pref_name__contains=kinase',
        'molecular_weight_filter': 'molecule?molecule_properties__mw_freebase__lte=300',
        'binding_assays': 'assay?assay_type=B',
        'similarity_search': 'similarity/CC(=O)Oc1ccccc1C(=O)O/80',
        'molecule_image': 'image/CHEMBL25.svg'
    }
}

with open('chembl.pkl', 'wb') as f:
    pickle.dump(chembl_schema, f)

print('ChEMBL schema created successfully')
