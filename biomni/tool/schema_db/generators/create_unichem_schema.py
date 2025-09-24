import pickle

# UniChem schema based on official API documentation
unichem_schema = {
    "base_url": "https://www.ebi.ac.uk/unichem/api/v1",
    "description": "UniChem 2.0 REST API for chemical cross-references",
    "docs_url": "https://chembl.gitbook.io/unichem/api/",
    "license": "CC0 (service/data)",
    "endpoints": {
        "sources": {
            "description": "Get information about available data sources",
            "url": "/sources",
            "method": "GET",
            "parameters": {"source_id": "Optional: Specific source ID to get details for (e.g., /sources/1)"},
        },
        "compounds": {
            "description": "Search for compounds by InChI Key, InChI, source compound ID, or UCI",
            "url": "/compounds",
            "method": "POST",
            "parameters": {
                "type": "Type of identifier: uci, inchi, inchikey, or sourceID",
                "compound": "Compound representation (ignored when type is sourceID)",
                "sourceID": "Unique ID on the source database (only when type is sourceID)",
            },
        },
        "connectivity": {
            "description": "Find compounds by InChI connectivity layers (formula, connections, H atoms, charge)",
            "url": "/connectivity",
            "method": "POST",
            "parameters": {
                "type": "Type of identifier: uci, inchi, inchikey, or sourceID",
                "compound": "Compound representation (ignored when type is sourceID)",
                "sourceID": "Unique ID on the source database (only when type is sourceID)",
                "searchComponents": "Boolean: whether to use individual components of mixtures (default: false)",
            },
        },
    },
    "identifier_types": [
        "uci",  # Unique Compound ID
        "inchi",  # InChI string
        "inchikey",  # InChI Key
        "sourceID",  # Source compound ID
    ],
    "common_sources": {
        "1": "ChEMBL",
        "2": "DrugBank",
        "3": "PDB",
        "4": "GToPdb",
        "5": "PubChem",
        "6": "KEGG",
        "7": "ChEBI",
        "8": "NIH Clinical Collection",
        "9": "ZINC",
        "10": "eMolecules",
        "11": "IBM Patent System",
        "12": "FDA Orange Book",
        "13": "IBM Strategic IP Insight Platform",
        "14": "Atlas",
        "15": "Patentscope",
        "16": "Binding Database",
        "17": "EPA DSSTox",
        "18": "LINCS",
        "19": "Human Metabolome Database",
        "20": "BRENDA",
        "21": "Rhea",
        "22": "LIPID MAPS",
        "23": "CAS",
        "24": "ChemSpider",
        "25": "Pharmgkb",
        "26": "IUPHAR/BPS Guide to PHARMACOLOGY",
        "27": "MCULE",
        "28": "NMRSHIFTDB2",
        "29": "LINCS",
        "30": "ACToR",
        "31": "EPA CompTox Chemistry Dashboard",
        "32": "NORMAN Suspect List Exchange",
        "33": "PubChem",
        "34": "COCONUT",
        "35": "ChemicalBook",
        "36": "Metabolights",
        "37": "SURECHEMBL",
        "38": "PubChem",
        "39": "ChemIDplus",
        "40": "Selleck",
        "41": "Cayman Chemical",
        "42": "MolPort",
        "43": "Nikkaji",
        "44": "Reaxys",
        "45": "ZINC15",
        "46": "eMolecules Plus",
        "47": "Mcule",
        "48": "Molbase",
        "49": "ChemSpace",
        "50": "Amadis Chemical",
        "51": "AKos Consulting & Solutions",
        "52": "Princeton BioMolecular Research",
        "53": "Specs",
        "54": "TargetMol",
        "55": "TimTec",
    },
    "examples": {
        "search_by_inchikey": {
            "endpoint": "/compounds",
            "method": "POST",
            "data": {"type": "inchikey", "compound": "LMXNVOREDXZICN-WDSOQIARSA-N"},
        },
        "search_by_inchi": {
            "endpoint": "/compounds",
            "method": "POST",
            "data": {
                "type": "inchi",
                "compound": "InChI=1S/C7H8N4O2/c1-10-5-4(8-3-9-5)6(12)11(2)7(10)13/h3H,1-2H3,(H,8,9)",
            },
        },
        "search_by_source_id": {
            "endpoint": "/compounds",
            "method": "POST",
            "data": {"type": "sourceID", "sourceID": "CHEMBL25"},
        },
        "connectivity_search": {
            "endpoint": "/connectivity",
            "method": "POST",
            "data": {
                "type": "inchi",
                "compound": "InChI=1S/C7H8N4O2.C2H7NO/c1-10-5-4(8-3-9-5)6(12)11(2)7(10)13;3-1-2-4/h3H,1-2H3,(H,8,9);4H,1-3H2",
                "searchComponents": True,
            },
        },
        "get_all_sources": {"endpoint": "/sources", "method": "GET"},
        "get_specific_source": {"endpoint": "/sources/1", "method": "GET"},
    },
}

with open("unichem.pkl", "wb") as f:
    pickle.dump(unichem_schema, f)

print("UniChem schema created successfully")
