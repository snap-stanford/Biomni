import pickle

# UniChem schema
unichem_schema = {
    'base_url': 'https://www.ebi.ac.uk/unichem/beta/api/v1',
    'description': 'UniChem 2.0 REST API for chemical cross-references',
    'docs_url': 'https://www.ebi.ac.uk/unichem/api/docs',
    'license': 'CC0 (service/data)',
    'method': 'POST',
    
    'endpoints': {
        'compounds': {
            'description': 'Search for compounds by InChI Key, SMILES, or other identifiers',
            'url': '/compounds',
            'method': 'POST',
            'parameters': {
                'compound': 'The compound identifier (InChI Key, SMILES, etc.)',
                'sourceID': 'Source database ID (1=ChEMBL, 2=DrugBank, etc.)',
                'type': 'Type of identifier (inchikey, smiles, etc.)'
            }
        },
        'sources': {
            'description': 'Get information about available data sources',
            'url': '/sources',
            'method': 'GET'
        }
    },
    
    'identifier_types': [
        'inchikey',
        'smiles',
        'inchi'
    ],
    
    'common_sources': {
        '1': 'ChEMBL',
        '2': 'DrugBank',
        '3': 'PDB',
        '4': 'GToPdb',
        '5': 'PubChem',
        '6': 'KEGG',
        '7': 'ChEBI',
        '8': 'NIH Clinical Collection',
        '9': 'ZINC',
        '10': 'eMolecules',
        '11': 'IBM Patent System',
        '12': 'FDA Orange Book',
        '13': 'IBM Strategic IP Insight Platform',
        '14': 'Atlas',
        '15': 'Patentscope',
        '16': 'Binding Database',
        '17': 'EPA DSSTox',
        '18': 'LINCS',
        '19': 'Human Metabolome Database',
        '20': 'BRENDA',
        '21': 'Rhea',
        '22': 'LIPID MAPS',
        '23': 'CAS',
        '24': 'ChemSpider',
        '25': 'Pharmgkb',
        '26': 'IUPHAR/BPS Guide to PHARMACOLOGY',
        '27': 'MCULE',
        '28': 'NMRSHIFTDB2',
        '29': 'LINCS',
        '30': 'ACToR',
        '31': 'EPA CompTox Chemistry Dashboard',
        '32': 'NORMAN Suspect List Exchange',
        '33': 'PubChem',
        '34': 'COCONUT',
        '35': 'ChemicalBook',
        '36': 'Metabolights',
        '37': 'SURECHEMBL',
        '38': 'PubChem',
        '39': 'ChemIDplus',
        '40': 'Selleck',
        '41': 'Cayman Chemical',
        '42': 'MolPort',
        '43': 'Nikkaji',
        '44': 'Reaxys',
        '45': 'ZINC15',
        '46': 'eMolecules Plus',
        '47': 'Mcule',
        '48': 'Molbase',
        '49': 'ChemSpace',
        '50': 'Amadis Chemical',
        '51': 'AKos Consulting & Solutions',
        '52': 'Princeton BioMolecular Research',
        '53': 'Specs',
        '54': 'TargetMol',
        '55': 'TimTec'
    },
    
    'examples': {
        'search_by_inchikey': {
            'endpoint': '/compounds',
            'method': 'POST',
            'data': {
                'compound': 'RYYVLZVUVIJVGH-UHFFFAOYSA-N',
                'sourceID': 1,
                'type': 'inchikey'
            }
        },
        'search_by_smiles': {
            'endpoint': '/compounds',
            'method': 'POST',
            'data': {
                'compound': 'CC(=O)OC1=CC=CC=C1C(=O)O',
                'sourceID': 1,
                'type': 'smiles'
            }
        },
        'get_sources': {
            'endpoint': '/sources',
            'method': 'GET'
        }
    }
}

with open('unichem.pkl', 'wb') as f:
    pickle.dump(unichem_schema, f)

print('UniChem schema created successfully')
