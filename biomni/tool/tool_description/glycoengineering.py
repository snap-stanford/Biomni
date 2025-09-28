description = [
    {
        "name": "find_n_glycosylation_motifs",
        "description": "Scan a protein sequence for canonical N-glycosylation sequons (N-X-[S/T], X≠P). Returns positions and motifs.",
        "required_parameters": [
            {
                "name": "sequence",
                "type": "str",
                "description": "Protein sequence (one-letter AA codes)",
                "default": None,
            }
        ],
        "optional_parameters": [
            {
                "name": "allow_overlap",
                "type": "bool",
                "description": "Allow overlapping motif detections",
                "default": False,
            }
        ],
    },
    {
        "name": "predict_o_glycosylation_hotspots",
        "description": "Heuristic O-glycosite hotspot prediction using local S/T density (lightweight baseline; see NetOGlyc 4.0 for SOTA).",
        "required_parameters": [
            {
                "name": "sequence",
                "type": "str",
                "description": "Protein sequence (one-letter AA codes)",
                "default": None,
            }
        ],
        "optional_parameters": [
            {"name": "window", "type": "int", "description": "Odd-sized window for local density (>=3)", "default": 7},
            {
                "name": "min_st_fraction",
                "type": "float",
                "description": "Min S/T fraction in window to flag site",
                "default": 0.4,
            },
            {
                "name": "disallow_proline_next",
                "type": "bool",
                "description": "Avoid S/T immediately followed by Proline",
                "default": True,
            },
        ],
    },
    {
        "name": "list_glycoengineering_resources",
        "description": "Curated list of external glycoengineering tools and resources (links and notes) as referenced in issue #198.",
        "required_parameters": [],
        "optional_parameters": [],
    },
]
