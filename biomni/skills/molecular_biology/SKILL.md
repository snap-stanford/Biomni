---
name: Molecular Biology
description: Molecular biology workflows including CRISPR design, sgRNA libraries, and sequence analysis.
---

## Tools
- **annotate_open_reading_frames**: Find all Open Reading Frames (ORFs) in a DNA sequence using Biopython, searching both forward and reverse complement strands.
- **annotate_plasmid**: Annotate a DNA sequence using pLannotate's command-line interface.
- **get_gene_coding_sequence**: Retrieves the coding sequence(s) of a specified gene from NCBI Entrez.
- **get_plasmid_sequence**: Unified function to retrieve plasmid sequences from either Addgene or NCBI. If is_addgene is True or identifier is numeric, uses Addgene. Otherwise searches NCBI using the plasmid name.
- **align_sequences**: Align short sequences (primers) to a longer sequence, allowing for one mismatch. Checks both forward and reverse complement strands.
- **pcr_simple**: Simulate PCR amplification with given primers and sequence.
- **digest_sequence**: Simulates restriction enzyme digestion of a DNA sequence and returns the resulting fragments with their properties.
- **find_restriction_sites**: Identifies restriction enzyme sites in a given DNA sequence for specified enzymes.
- **find_restriction_enzymes**: Finds common restriction enzyme sites in a DNA sequence and returns their cut positions.
- **find_sequence_mutations**: Compare query sequence against reference sequence to identify mutations.
- **design_knockout_sgrna**: Design sgRNAs for CRISPR knockout by searching pre-computed sgRNA libraries. Returns optimized guide RNAs for targeting a specific gene.
- **get_oligo_annealing_protocol**: Return a standard protocol for annealing oligonucleotides without phosphorylation.
- **get_golden_gate_assembly_protocol**: Return a customized protocol for Golden Gate assembly based on the number of inserts and specific DNA sequences.
- **get_bacterial_transformation_protocol**: Return a standard protocol for bacterial transformation.
- **design_primer**: Design a single primer within the given sequence window.
- **design_verification_primers**: Design Sanger sequencing primers to verify a specific region in a plasmid. First tries to use primers from an existing primer pool. If they cannot fully cover the region, designs additional primers as needed.
- **design_golden_gate_oligos**: Design complementary oligonucleotides with Type IIS restriction enzyme overhangs for Golden Gate assembly based on restriction site analysis of the backbone.
- **golden_gate_assembly**: Simulate Golden Gate assembly to predict final construct sequences from backbone and fragment sequences.
