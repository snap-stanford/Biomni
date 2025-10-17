description = [
    {
        "description": "Find all Open Reading Frames (ORFs) in a DNA sequence using "
        "Biopython, searching both forward and reverse complement "
        "strands.",
        "name": "annotate_open_reading_frames",
        "optional_parameters": [
            {
                "default": False,
                "description": "Whether to search the reverse complement strand",
                "name": "search_reverse",
                "type": "bool",
            },
            {
                "default": False,
                "description": "Whether to filter out ORFs with same end but later start",
                "name": "filter_subsets",
                "type": "bool",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "DNA sequence to analyze",
                "name": "sequence",
                "type": "str",
            },
            {
                "default": None,
                "description": "Minimum length of ORF in nucleotides",
                "name": "min_length",
                "type": "int",
            },
        ],
    },
    {
        "description": "Annotate a DNA sequence using pLannotate's command-line interface.",
        "name": "annotate_plasmid",
        "optional_parameters": [
            {
                "default": True,
                "description": "Whether the sequence is circular",
                "name": "is_circular",
                "type": "bool",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The DNA sequence to annotate",
                "name": "sequence",
                "type": "str",
            }
        ],
    },
    {
        "description": "Retrieves the coding sequence(s) of a specified gene from NCBI Entrez.",
        "name": "get_gene_coding_sequence",
        "optional_parameters": [
            {
                "default": None,
                "description": "Email address for NCBI Entrez (recommended)",
                "name": "email",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Name of the gene",
                "name": "gene_name",
                "type": "str",
            },
            {
                "default": None,
                "description": "Name of the organism",
                "name": "organism",
                "type": "str",
            },
        ],
    },
    {
        "description": "Unified function to retrieve plasmid sequences from either "
        "Addgene or NCBI. If is_addgene is True or identifier is "
        "numeric, uses Addgene. Otherwise searches NCBI using the "
        "plasmid name.",
        "name": "get_plasmid_sequence",
        "optional_parameters": [
            {
                "default": None,
                "description": "Force Addgene lookup if True, force "
                "NCBI if False. If None, attempts to "
                "auto-detect based on identifier "
                "format.",
                "name": "is_addgene",
                "type": "bool",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Either an Addgene ID or plasmid name",
                "name": "identifier",
                "type": "str",
            }
        ],
    },
    {
        "description": "Align short sequences (primers) to a longer sequence, "
        "allowing for one mismatch. Checks both forward and reverse "
        "complement strands.",
        "name": "align_sequences",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "Target DNA sequence",
                "name": "long_seq",
                "type": "str",
            },
            {
                "default": None,
                "description": "Single primer or list of primers",
                "name": "short_seqs",
                "type": "string",
            },
        ],
    },
    {
        "description": "Simulate PCR amplification with given primers and sequence.",
        "name": "pcr_simple",
        "optional_parameters": [
            {
                "default": False,
                "description": "Whether the sequence is circular",
                "name": "circular",
                "type": "bool",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Either a sequence string or path to plasmid file",
                "name": "sequence",
                "type": "str",
            },
            {
                "default": None,
                "description": "Forward primer sequence (5' to 3')",
                "name": "forward_primer",
                "type": "str",
            },
            {
                "default": None,
                "description": "Reverse primer sequence (5' to 3')",
                "name": "reverse_primer",
                "type": "str",
            },
        ],
    },
    {
        "description": "Simulates restriction enzyme digestion of a DNA sequence and "
        "returns the resulting fragments with their properties.",
        "name": "digest_sequence",
        "optional_parameters": [
            {
                "default": True,
                "description": "Whether the DNA sequence is circular (True) or linear (False)",
                "name": "is_circular",
                "type": "bool",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Input DNA sequence to be digested",
                "name": "dna_sequence",
                "type": "str",
            },
            {
                "default": None,
                "description": "Names of restriction enzymes to use for digestion",
                "name": "enzyme_names",
                "type": "array",
                "items": {"type": "string"},
            },
        ],
    },
    {
        "description": "Identifies restriction enzyme sites in a given DNA sequence for specified enzymes.",
        "name": "find_restriction_sites",
        "optional_parameters": [
            {
                "default": True,
                "description": "Whether the DNA sequence is circular (True) or linear (False)",
                "name": "is_circular",
                "type": "bool",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Complete input DNA sequence",
                "name": "dna_sequence",
                "type": "str",
            },
            {
                "default": None,
                "description": "List of restriction enzyme names to check",
                "name": "enzymes",
                "type": "array",
                "items": {"type": "string"},
            },
        ],
    },
    {
        "description": "Finds common restriction enzyme sites in a DNA sequence and returns their cut positions.",
        "name": "find_restriction_enzymes",
        "optional_parameters": [
            {
                "default": False,
                "description": "Whether the sequence is circular",
                "name": "is_circular",
                "type": "bool",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "DNA sequence to analyze",
                "name": "sequence",
                "type": "str",
            }
        ],
    },
    {
        "description": "Compare query sequence against reference sequence to identify mutations.",
        "name": "find_sequence_mutations",
        "optional_parameters": [
            {
                "default": 1,
                "description": "The start position of the query sequence",
                "name": "query_start",
                "type": "int",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The sequence being analyzed",
                "name": "query_sequence",
                "type": "str",
            },
            {
                "default": None,
                "description": "The reference sequence to compare against",
                "name": "reference_sequence",
                "type": "str",
            },
        ],
    },
    {
        "description": "Design sgRNAs for CRISPR knockout by searching pre-computed "
        "sgRNA libraries. Returns optimized guide RNAs for targeting "
        "a specific gene.",
        "name": "design_knockout_sgrna",
        "optional_parameters": [
            {
                "default": "human",
                "description": "Target organism species",
                "name": "species",
                "type": "str",
            },
            {
                "default": 1,
                "description": "Number of guides to return",
                "name": "num_guides",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": 'Target gene symbol/name (e.g., "EGFR", "TP53")',
                "name": "gene_name",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the data lake",
                "name": "data_lake_path",
                "type": "str",
            },
        ],
    },
    {
        "description": "Return a standard protocol for annealing oligonucleotides without phosphorylation.",
        "name": "get_oligo_annealing_protocol",
        "optional_parameters": [],
        "required_parameters": [],
    },
    {
        "description": "Return a customized protocol for Golden Gate assembly based "
        "on the number of inserts and specific DNA sequences.",
        "name": "get_golden_gate_assembly_protocol",
        "optional_parameters": [
            {
                "default": 1,
                "description": "Number of inserts to be assembled",
                "name": "num_inserts",
                "type": "int",
            },
            {
                "default": 75.0,
                "description": "Amount of vector to use in ng",
                "name": "vector_amount_ng",
                "type": "float",
            },
            {
                "default": None,
                "description": "List of insert lengths in bp",
                "name": "insert_lengths",
                "type": "array",
                "items": {"type": "integer"},
            },
            {
                "default": False,
                "description": "Whether this is for library preparation",
                "name": "is_library_prep",
                "type": "bool",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Type IIS restriction enzyme to be used",
                "name": "enzyme_name",
                "type": "str",
            },
            {
                "default": None,
                "description": "Length of the destination vector in bp",
                "name": "vector_length",
                "type": "int",
            },
        ],
    },
    {
        "description": "Return a standard protocol for bacterial transformation.",
        "name": "get_bacterial_transformation_protocol",
        "optional_parameters": [
            {
                "default": "ampicillin",
                "description": "Selection antibiotic",
                "name": "antibiotic",
                "type": "str",
            },
            {
                "default": False,
                "description": "Whether the sequence contains repetitive elements",
                "name": "is_repetitive",
                "type": "bool",
            },
        ],
        "required_parameters": [],
    },
    {
        "description": "Design a single primer within the given sequence window.",
        "name": "design_primer",
        "optional_parameters": [
            {
                "default": 20,
                "description": "Length of the primer to design",
                "name": "primer_length",
                "type": "int",
            },
            {
                "default": 0.4,
                "description": "Minimum GC content",
                "name": "min_gc",
                "type": "float",
            },
            {
                "default": 0.6,
                "description": "Maximum GC content",
                "name": "max_gc",
                "type": "float",
            },
            {
                "default": 55.0,
                "description": "Minimum melting temperature in °C",
                "name": "min_tm",
                "type": "float",
            },
            {
                "default": 65.0,
                "description": "Maximum melting temperature in °C",
                "name": "max_tm",
                "type": "float",
            },
            {
                "default": 100,
                "description": "Size of window to search for primers",
                "name": "search_window",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Target DNA sequence",
                "name": "sequence",
                "type": "str",
            },
            {
                "default": None,
                "description": "Starting position for primer search",
                "name": "start_pos",
                "type": "int",
            },
        ],
    },
    {
        "description": "Design Sanger sequencing primers to verify a specific region "
        "in a plasmid. First tries to use primers from an existing "
        "primer pool. If they cannot fully cover the region, designs "
        "additional primers as needed.",
        "name": "design_verification_primers",
        "optional_parameters": [
            {
                "default": None,
                "description": "List of existing primers with their sequences and optional names",
                "name": "existing_primers",
                "type": "array",
                "items": {"type": "object", "additionalProperties": {"type": "string"}},
            },
            {
                "default": True,
                "description": "Whether the plasmid is circular",
                "name": "is_circular",
                "type": "bool",
            },
            {
                "default": 800,
                "description": "Typical read length for each primer in base pairs",
                "name": "coverage_length",
                "type": "int",
            },
            {
                "default": 20,
                "description": "Length of newly designed primers",
                "name": "primer_length",
                "type": "int",
            },
            {
                "default": 0.4,
                "description": "Minimum GC content for new primers",
                "name": "min_gc",
                "type": "float",
            },
            {
                "default": 0.6,
                "description": "Maximum GC content for new primers",
                "name": "max_gc",
                "type": "float",
            },
            {
                "default": 55.0,
                "description": "Minimum melting temperature in °C",
                "name": "min_tm",
                "type": "float",
            },
            {
                "default": 65.0,
                "description": "Maximum melting temperature in °C",
                "name": "max_tm",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "The complete plasmid sequence",
                "name": "plasmid_sequence",
                "type": "str",
            },
            {
                "default": None,
                "description": "Start and end positions to verify (0-based indexing)",
                "name": "target_region",
                "type": "Tuple[int, int]",
            },
        ],
    },
    {
        "description": "Design complementary oligonucleotides with Type IIS "
        "restriction enzyme overhangs for Golden Gate assembly based "
        "on restriction site analysis of the backbone.",
        "name": "design_golden_gate_oligos",
        "optional_parameters": [
            {
                "default": True,
                "description": "Whether the backbone is circular",
                "name": "is_circular",
                "type": "bool",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Complete backbone sequence",
                "name": "backbone_sequence",
                "type": "str",
            },
            {
                "default": None,
                "description": "Sequence to be inserted (e.g., sgRNA target sequence)",
                "name": "insert_sequence",
                "type": "str",
            },
            {
                "default": None,
                "description": "Type IIS restriction enzyme to be used",
                "name": "enzyme_name",
                "type": "str",
            },
        ],
    },
    {
        "description": "Simulate Golden Gate assembly to predict final construct "
        "sequences from backbone and fragment sequences.",
        "name": "golden_gate_assembly",
        "optional_parameters": [
            {
                "default": True,
                "description": "Whether the backbone is circular",
                "name": "is_circular",
                "type": "bool",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Complete backbone sequence",
                "name": "backbone_sequence",
                "type": "str",
            },
            {
                "default": None,
                "description": 'Type IIS restriction enzyme to be used (e.g., "BsmBI", "BsaI")',
                "name": "enzyme_name",
                "type": "str",
            },
            {
                "default": None,
                "description": "List of fragments to insert, containing one of: "
                "name + fwd_oligo + rev_oligo (oligo pair with matching overhangs) or "
                "name + sequence (double-stranded DNA fragment containing restriction sites)",
                "name": "fragments",
                "type": "array",
                "items": {"type": "object", "additionalProperties": {"type": "string"}},
            },
        ],
    },
]
