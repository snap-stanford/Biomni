def liftover_coordinates(
    chromosome: str,
    position: int,
    input_format: str,
    output_format: str,
    data_path: str,
) -> str:
    """Perform liftover of genomic coordinates between hg19 and hg38 formats with detailed intermediate steps.

    Args:
        chromosome (str): Chromosome number (e.g., '1', 'X').
        position (int): Genomic position.
        input_format (str): Input genome build ('hg19' or 'hg38').
        output_format (str): Output genome build ('hg19' or 'hg38').
        data_path (str): Path to liftover chain files.

    Returns:
        str: A detailed string explaining the steps and the final result or any error encountered.

    """
    from pyliftover import LiftOver

    steps = []

    try:
        steps.append(
            f"Starting liftover process for chromosome {chromosome}, position {position} from {input_format} to {output_format}."
        )

        # Load the liftover chain files
        steps.append("Loading liftover chain files...")
        hg19_to_hg38_liftover = LiftOver(data_path + "/liftover/hg19ToHg38.over.chain.gz")
        hg38_to_hg19_liftover = LiftOver(data_path + "/liftover/hg38ToHg19.over.chain.gz")
        steps.append("Liftover chain files loaded successfully.")

        # Choose the appropriate LiftOver object
        if input_format == "hg19" and output_format == "hg38":
            lo = hg19_to_hg38_liftover
            steps.append("Selected liftover chain: hg19 to hg38.")
        elif input_format == "hg38" and output_format == "hg19":
            lo = hg38_to_hg19_liftover
            steps.append("Selected liftover chain: hg38 to hg19.")
        else:
            steps.append("Error: Unsupported format conversion.")
            return "\n".join(
                steps
                + ["Error: Unsupported format conversion. Supported formats are 'hg19' to 'hg38' or 'hg38' to 'hg19'."]
            )

        # Perform the liftover conversion
        steps.append(f"Performing liftover for chr{chromosome}, position {position}...")
        lifted_coordinates = lo.convert_coordinate(f"chr{chromosome}", position)

        if lifted_coordinates:
            result = (
                f"Successfully lifted coordinates from {input_format} to {output_format}.\n"
                f"Original: chr{chromosome}, position {position}.\n"
                f"Lifted: chromosome {lifted_coordinates[0][0]}, position {lifted_coordinates[0][1]}, strand {lifted_coordinates[0][2]}."
            )
            steps.append(result)
            return "\n".join(steps)
        else:
            steps.append("Error: Liftover failed. No coordinates found for the given input.")
            return "\n".join(steps + ["Error: Liftover failed. No coordinates found."])

    except Exception as e:
        steps.append(f"Exception encountered: {str(e)}")
        return "\n".join(steps)


import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn, optim


def bayesian_finemapping_with_deep_vi(
    gwas_summary_path,
    ld_matrix,
    n_iterations=5000,
    learning_rate=0.01,
    hidden_dim=64,
    credible_threshold=0.95,
):
    """Performs Bayesian fine-mapping from GWAS summary statistics using deep variational inference.

    This function implements a deep neural network-based variational inference approach to compute
    posterior inclusion probabilities (PIPs) and credible sets for putative causal variants from
    GWAS summary statistics and linkage disequilibrium (LD) information.

    Parameters
    ----------
    gwas_summary_path : str
        Path to CSV or TSV file containing GWAS summary statistics. Expected columns:
        - 'variant_id': Identifier for each variant
        - 'effect_size': Effect size (beta) for each variant
        - 'pvalue': P-value for each variant
        - 'se': Standard error for each variant (optional)

    ld_matrix : numpy.ndarray
        Linkage disequilibrium matrix with pairwise correlations between variants.

    n_iterations : int, optional
        Number of training iterations for the variational inference algorithm.
        Default is 5000.

    learning_rate : float, optional
        Learning rate for the optimization algorithm. Default is 0.01.

    hidden_dim : int, optional
        Hidden dimension size for the neural network. Default is 64.

    credible_threshold : float, optional
        Threshold for defining the credible set (e.g., 0.95 for a 95% credible set).
        Default is 0.95.

    Returns
    -------
    str
        A detailed research log of the fine-mapping analysis including:
        - Number of variants analyzed
        - Top variants ranked by posterior inclusion probability
        - Credible set variants
        - Visualizations of the posterior distributions

    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Initialize the research log
    log = []
    log.append(
        f"# Bayesian Fine-mapping Analysis with Deep Variational Inference - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    log.append("\n## Data Preprocessing")

    # Load data from file
    try:
        if gwas_summary_path.endswith(".csv"):
            gwas_summary = pd.read_csv(gwas_summary_path)
        elif gwas_summary_path.endswith((".tsv", ".txt")):
            gwas_summary = pd.read_csv(gwas_summary_path, sep="\t")
        else:
            log.append("Error: Unsupported file format. Please provide a CSV or TSV file.")
            return "\n".join(log)
        log.append(f"Successfully loaded GWAS summary data from {gwas_summary_path}")
    except Exception as e:
        log.append(f"Error loading GWAS summary data: {str(e)}")
        return "\n".join(log)

    # Check input data
    if gwas_summary is None:
        log.append("Error: Failed to load GWAS summary data.")
        return "\n".join(log)

    if ld_matrix is None:
        log.append("Error: LD matrix is required for fine-mapping analysis.")
        return "\n".join(log)

    n_variants = len(gwas_summary)
    log.append(f"Analyzing {n_variants} genetic variants")

    # Check if LD matrix dimensions match the number of variants
    if ld_matrix.shape[0] != n_variants or ld_matrix.shape[1] != n_variants:
        log.append(f"Error: LD matrix dimensions ({ld_matrix.shape}) do not match number of variants ({n_variants})")
        return "\n".join(log)

    # Prepare data for analysis
    log.append("\nPreparing data for analysis...")

    # Compute Z-scores if not already present
    if "z_score" not in gwas_summary.columns:
        log.append("Computing Z-scores from effect sizes and standard errors...")
        if "se" in gwas_summary.columns:
            gwas_summary["z_score"] = gwas_summary["effect_size"] / gwas_summary["se"]
        else:
            # Approximate Z-scores from p-values
            log.append("Standard errors not available, approximating Z-scores from p-values...")
            # Convert p-values to Z-scores (two-sided test)
            from scipy.stats import norm

            gwas_summary["z_score"] = (
                gwas_summary["effect_size"].abs()
                / gwas_summary["effect_size"]
                * norm.ppf(1 - gwas_summary["pvalue"] / 2)
            )

    # Convert data to tensors
    z_scores = torch.FloatTensor(gwas_summary["z_score"].values)
    ld_tensor = torch.FloatTensor(ld_matrix)

    log.append(f"Processed {len(z_scores)} z-scores from GWAS summary")
    log.append("LD matrix shape: " + str(ld_matrix.shape))

    # Define the variational inference model
    class VariationalFineMapping(nn.Module):
        def __init__(self, n_variants, hidden_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_variants, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            # Output log alpha parameters for the Bernoulli variables
            self.log_alpha = nn.Linear(hidden_dim, n_variants)

        def forward(self, x):
            h = self.encoder(x)
            log_alpha = self.log_alpha(h)
            # Apply sigmoid to get inclusion probabilities
            return torch.sigmoid(log_alpha)

        def elbo_loss(self, z_scores, ld_matrix, pips, n_samples=10):
            # Sample from approximate posterior
            samples = torch.bernoulli(pips.unsqueeze(0).repeat(n_samples, 1))

            # Prior term (sparsity prior)
            prior_term = -0.01 * torch.sum(pips)

            # Likelihood term
            likelihood_term = 0
            for s in samples:
                # Compute expected z-scores under the model
                expected_z = torch.matmul(ld_matrix, s * z_scores)
                # Compute likelihood
                likelihood_term += -torch.sum((z_scores - expected_z) ** 2)

            likelihood_term /= n_samples

            return -(prior_term + likelihood_term)

    # Initialize model, optimizer and training
    log.append("\n## Initializing deep variational inference model")
    model = VariationalFineMapping(n_variants, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    log.append("\n## Training variational inference model")
    losses = []

    for i in range(n_iterations):
        optimizer.zero_grad()
        pips = model(z_scores)
        loss = model.elbo_loss(z_scores, ld_tensor, pips)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (i + 1) % (n_iterations // 5) == 0:
            log.append(f"  Iteration {i + 1}/{n_iterations}, Loss: {loss.item():.4f}")

    # Get final posterior inclusion probabilities
    with torch.no_grad():
        final_pips = model(z_scores).numpy()

    # Create DataFrame with results
    results_df = gwas_summary.copy()
    results_df["pip"] = final_pips
    results_df = results_df.sort_values("pip", ascending=False)

    # Generate credible sets
    log.append("\n## Generating credible sets")

    # Sort variants by PIP
    sorted_variants = results_df.sort_values("pip", ascending=False)

    # Calculate cumulative sum of PIPs
    sorted_variants["cumulative_pip"] = sorted_variants["pip"].cumsum()

    # Identify variants in the credible set
    credible_set = sorted_variants[sorted_variants["cumulative_pip"] <= credible_threshold]

    if len(credible_set) == 0:
        # If no variants meet the threshold, include at least the top variant
        credible_set = sorted_variants.iloc[:1]

    log.append(f"Identified {len(credible_set)} variants in the {credible_threshold * 100}% credible set")

    # Save results to files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"finemapping_results_{timestamp}.csv"
    credible_set_file = f"credible_set_{timestamp}.csv"

    results_df.to_csv(results_file, index=False)
    credible_set.to_csv(credible_set_file, index=False)

    log.append(f"\nFull results saved to: {results_file}")
    log.append(f"Credible set saved to: {credible_set_file}")

    # Summary of top variants
    log.append("\n## Top variants by posterior inclusion probability (PIP)")
    for i, (_, row) in enumerate(results_df.head(10).iterrows()):
        log.append(f"  {i + 1}. Variant: {row['variant_id']}, PIP: {row['pip']:.4f}, P-value: {row['pvalue']:.2e}")

    log.append("\n## Variants in the credible set")
    for i, (_, row) in enumerate(credible_set.iterrows()):
        log.append(f"  {i + 1}. Variant: {row['variant_id']}, PIP: {row['pip']:.4f}")

    # Create a simple visualization
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(results_df[:50])), results_df["pip"][:50])
        plt.xlabel("Variant index (sorted by PIP)")
        plt.ylabel("Posterior Inclusion Probability")
        plt.title("Top 50 variants by PIP")
        plot_file = f"pip_plot_{timestamp}.png"
        plt.savefig(plot_file)
        plt.close()
        log.append(f"\nPlot of PIPs saved to: {plot_file}")
    except Exception as e:
        log.append(f"\nCould not create visualization: {str(e)}")

    log.append("\n## Analysis complete")

    return "\n".join(log)


def analyze_cas9_mutation_outcomes(
    reference_sequences,
    edited_sequences,
    cell_line_info=None,
    output_prefix="cas9_mutation_analysis",
):
    """Analyzes and categorizes mutations induced by Cas9 at target sites.

    Parameters
    ----------
    reference_sequences : dict
        Dictionary mapping sequence IDs to reference DNA sequences (strings)
    edited_sequences : dict of dict
        Nested dictionary: {sequence_id: {read_id: sequence}}
        Contains the edited/mutated sequences for each reference
    cell_line_info : dict, optional
        Dictionary mapping sequence IDs to cell line information (e.g., wildtype, knockout gene)
    output_prefix : str, optional
        Prefix for output files

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    from collections import defaultdict

    from Bio import pairwise2

    # Initialize results storage
    results = []
    mutation_counts = defaultdict(lambda: defaultdict(int))

    # Define mutation categories
    categories = {
        "no_mutation": "No mutation detected",
        "short_deletion": "Short deletion (1-10 bp)",
        "medium_deletion": "Medium deletion (11-30 bp)",
        "long_deletion": "Long deletion (>30 bp)",
        "single_insertion": "Single base insertion",
        "longer_insertion": "Longer insertion (>1 bp)",
        "indel": "Insertion and deletion",
    }

    log = "# Cas9-Induced Mutation Outcome Analysis\n\n"
    log += "## Analysis Steps:\n\n"
    log += "1. Loading and processing sequence data\n"
    log += f"2. Analyzing {len(reference_sequences)} target sites\n"

    # Process each reference sequence and its edited versions
    for seq_id, ref_seq in reference_sequences.items():
        cell_line = cell_line_info.get(seq_id, "Unknown") if cell_line_info else "Unknown"
        log += f"\n### Processing target site: {seq_id} (Cell line: {cell_line})\n"

        site_results = []
        site_mutation_counts = defaultdict(int)
        total_reads = len(edited_sequences.get(seq_id, {}))

        if total_reads == 0:
            log += f"No edited sequences found for {seq_id}\n"
            continue

        log += f"Analyzing {total_reads} sequence reads...\n"

        # Process each edited sequence for this reference
        for read_id, edited_seq in edited_sequences.get(seq_id, {}).items():
            # Perform sequence alignment
            alignments = pairwise2.align.globalms(ref_seq, edited_seq, 2, -1, -2, -0.5)

            if not alignments:
                log += f"Warning: Could not align read {read_id}\n"
                continue

            best_alignment = alignments[0]
            ref_aligned, edited_aligned, score, start, end = best_alignment

            # Analyze mutations
            deletions = []
            insertions = []
            del_count = 0
            ins_count = 0

            i, j = 0, 0
            while i < len(ref_aligned) and j < len(edited_aligned):
                if ref_aligned[i] == "-":  # Insertion in edited sequence
                    ins_start = j
                    while i < len(ref_aligned) and ref_aligned[i] == "-":
                        i += 1
                        j += 1
                    insertions.append((ins_start, j - ins_start))
                    ins_count += j - ins_start
                elif edited_aligned[j] == "-":  # Deletion in edited sequence
                    del_start = i
                    while j < len(edited_aligned) and edited_aligned[j] == "-":
                        i += 1
                        j += 1
                    deletions.append((del_start, i - del_start))
                    del_count += i - del_start
                else:
                    i += 1
                    j += 1

            # Categorize mutation
            mutation_type = "no_mutation"
            if del_count > 0 and ins_count > 0:
                mutation_type = "indel"
            elif del_count > 0:
                if del_count <= 10:
                    mutation_type = "short_deletion"
                elif del_count <= 30:
                    mutation_type = "medium_deletion"
                else:
                    mutation_type = "long_deletion"
            elif ins_count > 0:
                mutation_type = "single_insertion" if ins_count == 1 else "longer_insertion"

            # Add to results
            site_results.append(
                {
                    "sequence_id": seq_id,
                    "read_id": read_id,
                    "cell_line": cell_line,
                    "mutation_type": mutation_type,
                    "deletion_count": del_count,
                    "insertion_count": ins_count,
                }
            )

            site_mutation_counts[mutation_type] += 1
            mutation_counts[cell_line][mutation_type] += 1

        # Calculate percentages for this site
        log += "\nMutation distribution for this target site:\n"
        for mut_type, count in site_mutation_counts.items():
            percentage = (count / total_reads) * 100
            log += f"- {categories[mut_type]}: {count} reads ({percentage:.1f}%)\n"

        # Add site results to overall results
        results.extend(site_results)

    # Create results dataframe and save to CSV
    results_df = pd.DataFrame(results)
    output_file = f"{output_prefix}_detailed_results.csv"
    results_df.to_csv(output_file, index=False)

    # Create summary dataframe
    summary_data = []
    for cell_line, mut_counts in mutation_counts.items():
        total = sum(mut_counts.values())
        for mut_type, count in mut_counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            summary_data.append(
                {
                    "cell_line": cell_line,
                    "mutation_type": mut_type,
                    "count": count,
                    "percentage": percentage,
                }
            )

    summary_df = pd.DataFrame(summary_data)
    summary_file = f"{output_prefix}_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    # Add summary to log
    log += "\n## Overall Results Summary\n\n"
    log += f"Total sequences analyzed: {len(results)}\n"
    log += f"Detailed results saved to: {output_file}\n"
    log += f"Summary results saved to: {summary_file}\n\n"

    if cell_line_info:
        log += "### Mutation Distribution by Cell Line\n\n"
        for cell_line, mut_counts in mutation_counts.items():
            total = sum(mut_counts.values())
            if total == 0:
                continue

            log += f"#### {cell_line}\n"
            for mut_type, count in sorted(mut_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100
                log += f"- {categories[mut_type]}: {count} ({percentage:.1f}%)\n"
            log += "\n"

    return log


def analyze_crispr_genome_editing(original_sequence, edited_sequence, guide_rna, repair_template=None):
    """Analyzes CRISPR-Cas9 genome editing results by comparing original and edited sequences.

    Parameters
    ----------
    original_sequence : str
        The original DNA sequence before CRISPR-Cas9 editing
    edited_sequence : str
        The DNA sequence after CRISPR-Cas9 editing
    guide_rna : str
        The CRISPR guide RNA (crRNA) sequence used for targeting
    repair_template : str, optional
        The homology-directed repair template sequence, if used

    Returns
    -------
    str
        A research log summarizing the CRISPR-Cas9 editing analysis, including identified
        mutations and characterization of the edited loci

    """
    import datetime

    from Bio import pairwise2
    from Bio.Seq import Seq

    log = []
    log.append(f"CRISPR-Cas9 Genome Editing Analysis - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append("=" * 70)

    # Step 1: Find the target site in the original sequence
    log.append("\n1. Identifying target site in original sequence")
    target_site = original_sequence.find(guide_rna)
    if target_site == -1:
        # Try with the reverse complement
        guide_rna_seq = Seq(guide_rna)
        rev_comp_guide = str(guide_rna_seq.reverse_complement())
        target_site = original_sequence.find(rev_comp_guide)
        if target_site != -1:
            log.append(f"   - Target site found at position {target_site} (using reverse complement of guide RNA)")
            guide_rna = rev_comp_guide
        else:
            log.append("   - Warning: Guide RNA sequence not found in original sequence")
            target_site = None
    else:
        log.append(f"   - Target site found at position {target_site}")

    # Step 2: Align sequences to identify mutations
    log.append("\n2. Aligning original and edited sequences to identify mutations")
    alignments = pairwise2.align.globalms(original_sequence, edited_sequence, 2, -1, -2, -0.5)
    best_alignment = alignments[0]

    # Extract aligned sequences
    aligned_orig = best_alignment[0]
    aligned_edit = best_alignment[1]

    # Find mutations
    mutations = []
    indels = []

    for i in range(len(aligned_orig)):
        if aligned_orig[i] != aligned_edit[i]:
            orig_base = aligned_orig[i]
            edit_base = aligned_edit[i]

            # Skip gaps in counting the actual position
            actual_pos = len(aligned_orig[:i].replace("-", ""))

            if orig_base == "-":  # Insertion
                indels.append(f"Insertion of {edit_base} at position {actual_pos}")
            elif edit_base == "-":  # Deletion
                indels.append(f"Deletion of {orig_base} at position {actual_pos}")
            else:  # Substitution
                mutations.append(f"{orig_base}→{edit_base} at position {actual_pos}")

    # Log mutations
    if mutations:
        log.append("   - Substitutions detected:")
        for mutation in mutations:
            log.append(f"     * {mutation}")
    else:
        log.append("   - No substitutions detected")

    if indels:
        log.append("   - Insertions/Deletions detected:")
        for indel in indels:
            log.append(f"     * {indel}")
    else:
        log.append("   - No insertions or deletions detected")

    # Step 3: Check if editing occurred near the target site
    if target_site is not None:
        log.append("\n3. Analyzing mutations relative to target site")
        target_end = target_site + len(guide_rna)
        target_region = range(target_site - 3, target_end + 3)  # Include some buffer

        on_target_edits = [m for m in mutations + indels if any(str(pos) in m for pos in target_region)]

        if on_target_edits:
            log.append("   - On-target edits detected near guide RNA binding site:")
            for edit in on_target_edits:
                log.append(f"     * {edit}")
        else:
            log.append("   - No edits detected near guide RNA binding site")

    # Step 4: Check for homology-directed repair if template was provided
    if repair_template:
        log.append("\n4. Checking for homology-directed repair template incorporation")
        # Look for unique sequence markers from the repair template
        template_len = len(repair_template)
        marker_size = min(10, template_len // 3)
        marker = repair_template[template_len // 2 - marker_size // 2 : template_len // 2 + marker_size // 2]

        if marker in edited_sequence and marker not in original_sequence:
            log.append(f"   - Repair template marker '{marker}' found in edited sequence")
            log.append("   - Homology-directed repair likely successful")
        else:
            log.append("   - No clear evidence of repair template incorporation")
            log.append("   - Editing likely resulted from non-homologous end joining (NHEJ)")

    # Step 5: Overall assessment
    log.append("\n5. Overall assessment")
    if mutations or indels:
        log.append("   - CRISPR-Cas9 editing appears successful")
        if target_site is not None and any(
            str(pos) in "".join(mutations + indels) for pos in range(target_site, target_site + len(guide_rna))
        ):
            log.append("   - Edits occurred at the intended target site")
        else:
            log.append("   - Edits may have occurred outside the intended target site")
    else:
        log.append("   - No significant editing detected, CRISPR-Cas9 may not have been effective")

    return "\n".join(log)


def simulate_demographic_history(
    num_samples=10,
    sequence_length=100000,
    recombination_rate=1e-8,
    mutation_rate=1e-8,
    demographic_model="constant",
    demographic_params=None,
    coalescent_model="kingman",
    beta_coalescent_param=None,
    random_seed=None,
    output_file="simulated_sequences.vcf",
):
    """Simulate DNA sequences with specified demographic and coalescent histories using msprime.

    Parameters
    ----------
    num_samples : int
        Number of sample sequences to simulate
    sequence_length : int
        Length of the simulated sequence in base pairs
    recombination_rate : float
        Per-base recombination rate
    mutation_rate : float
        Per-base mutation rate
    demographic_model : str
        Type of demographic model to simulate. Options:
        - "constant": Constant population size
        - "bottleneck": Population bottleneck
        - "expansion": Population expansion
        - "contraction": Population contraction
        - "sawtooth": Sawtooth pattern of population size changes
    demographic_params : dict
        Parameters specific to the chosen demographic model. Supported formats::

            - For "constant": {"N": population size}
            - For "bottleneck": {
                "N_initial": initial pop size,
                "N_bottleneck": bottleneck pop size,
                "T_bottleneck": time of bottleneck (generations ago),
                "T_recovery": time of recovery (generations ago)
            }
            - For "expansion": {"N_initial": initial pop size, "N_final": final pop size, "T_expansion": time of expansion (generations ago)}
            - For "contraction": {"N_initial": initial pop size, "N_final": final pop size, "T_contraction": time of contraction (generations ago)}
            - For "sawtooth": {"N_values": list of population sizes, "times": list of times for changes}
    coalescent_model : str
        Type of coalescent model to use. Options:
        - "kingman": Standard Kingman coalescent
        - "beta": Beta-coalescent model
    beta_coalescent_param : float
        Parameter for beta-coalescent model (required if coalescent_model="beta")
    random_seed : int
        Seed for random number generator (for reproducibility)
    output_file : str
        Filename to save the simulated sequences (VCF format)

    Returns
    -------
    str
        Research log summarizing the simulation parameters and results

    """
    import time

    import msprime

    start_time = time.time()
    log = []
    log.append("Demographic History Simulation using msprime")
    log.append("=============================================")
    log.append("Parameters:")
    log.append(f"  - Number of samples: {num_samples}")
    log.append(f"  - Sequence length: {sequence_length} bp")
    log.append(f"  - Recombination rate: {recombination_rate}")
    log.append(f"  - Mutation rate: {mutation_rate}")
    log.append(f"  - Demographic model: {demographic_model}")
    log.append(f"  - Coalescent model: {coalescent_model}")

    # Set up demographic model
    if demographic_params is None:
        demographic_params = {"N": 10000}  # Default to constant population of 10000

    demography = msprime.Demography()
    demography.add_population(name="pop0", initial_size=1000)  # Default initial population

    if demographic_model == "constant":
        N = demographic_params.get("N", 10000)
        demography.add_population(name="pop0", initial_size=N)
        log.append(f"  - Constant population size: N = {N}")

    elif demographic_model == "bottleneck":
        N_initial = demographic_params.get("N_initial", 10000)
        N_bottleneck = demographic_params.get("N_bottleneck", 1000)
        T_bottleneck = demographic_params.get("T_bottleneck", 1000)
        T_recovery = demographic_params.get("T_recovery", 500)

        demography = msprime.Demography()
        demography.add_population(name="pop0", initial_size=N_initial)
        demography.add_population_parameters_change(time=T_recovery, initial_size=N_bottleneck)
        demography.add_population_parameters_change(time=T_bottleneck, initial_size=N_initial)

        log.append("  - Bottleneck model:")
        log.append(f"    * Initial population size: {N_initial}")
        log.append(f"    * Bottleneck population size: {N_bottleneck}")
        log.append(f"    * Bottleneck time (generations ago): {T_bottleneck}")
        log.append(f"    * Recovery time (generations ago): {T_recovery}")

    elif demographic_model == "expansion":
        N_initial = demographic_params.get("N_initial", 1000)
        N_final = demographic_params.get("N_final", 10000)
        T_expansion = demographic_params.get("T_expansion", 1000)

        demography = msprime.Demography()
        demography.add_population(name="pop0", initial_size=N_final)
        demography.add_population_parameters_change(time=T_expansion, initial_size=N_initial)

        log.append("  - Expansion model:")
        log.append(f"    * Initial population size: {N_initial}")
        log.append(f"    * Final population size: {N_final}")
        log.append(f"    * Expansion time (generations ago): {T_expansion}")

    elif demographic_model == "contraction":
        N_initial = demographic_params.get("N_initial", 10000)
        N_final = demographic_params.get("N_final", 1000)
        T_contraction = demographic_params.get("T_contraction", 1000)

        demography = msprime.Demography()
        demography.add_population(name="pop0", initial_size=N_final)
        demography.add_population_parameters_change(time=T_contraction, initial_size=N_initial)

        log.append("  - Contraction model:")
        log.append(f"    * Initial population size: {N_initial}")
        log.append(f"    * Final population size: {N_final}")
        log.append(f"    * Contraction time (generations ago): {T_contraction}")

    elif demographic_model == "sawtooth":
        N_values = demographic_params.get("N_values", [10000, 5000, 15000, 7500])
        times = demographic_params.get("times", [500, 1000, 1500])

        if len(N_values) != len(times) + 1:
            raise ValueError("For sawtooth model, N_values should have one more element than times")

        demography = msprime.Demography()
        demography.add_population(name="pop0", initial_size=N_values[0])

        for i, time in enumerate(times):
            demography.add_population_parameters_change(time=time, initial_size=N_values[i + 1])

        log.append("  - Sawtooth model:")
        log.append(f"    * Population sizes: {N_values}")
        log.append(f"    * Change times (generations ago): {times}")

    else:
        raise ValueError(f"Unknown demographic model: {demographic_model}")

    # Set up coalescent model
    model = None
    if coalescent_model == "kingman":
        model = msprime.StandardCoalescent()
        log.append("  - Using standard Kingman coalescent")
    elif coalescent_model == "beta":
        if beta_coalescent_param is None:
            beta_coalescent_param = 1.5
        model = msprime.BetaCoalescent(alpha=beta_coalescent_param)
        log.append(f"  - Using Beta-coalescent with alpha = {beta_coalescent_param}")
    else:
        raise ValueError(f"Unknown coalescent model: {coalescent_model}")

    # Run simulation
    log.append("\nRunning simulation...")

    ts = msprime.sim_ancestry(
        samples=num_samples,
        recombination_rate=recombination_rate,
        sequence_length=sequence_length,
        demography=demography,
        model=model,
        random_seed=random_seed,
    )

    # Add mutations
    mts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=random_seed)

    # Save to VCF
    with open(output_file, "w") as vcf_file:
        mts.write_vcf(vcf_file)

    # Calculate some basic statistics
    diversity = mts.diversity()
    num_sites = mts.num_sites
    num_trees = mts.num_trees

    # Log results
    end_time = time.time()
    runtime = end_time - start_time

    log.append(f"Simulation completed in {runtime:.2f} seconds")
    log.append("\nResults:")
    log.append(f"  - Number of segregating sites: {num_sites}")
    log.append(f"  - Number of trees in ARG: {num_trees}")
    log.append(f"  - Nucleotide diversity (π): {diversity:.6f}")
    log.append(f"  - Output saved to: {os.path.abspath(output_file)}")

    return "\n".join(log)


def identify_transcription_factor_binding_sites(sequence, tf_name, threshold=0.8, output_file=None):
    """Identifies binding sites for a specific transcription factor in a genomic sequence.

    Parameters
    ----------
    sequence : str
        The genomic DNA sequence to analyze
    tf_name : str
        Name of the transcription factor to search for (e.g., 'Hsf1', 'GATA1')
    threshold : float, optional
        Minimum score threshold for reporting binding sites (0.0-1.0, default: 0.8)
    output_file : str, optional
        Path to save the results (default: None, results only in log)

    Returns
    -------
    str
        Research log detailing the binding site identification process and results

    """
    import datetime
    import io

    import requests
    from Bio import motifs
    from Bio.Seq import Seq

    log = f"# Transcription Factor Binding Site Analysis: {tf_name}\n"
    log += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Step 1: Get the PWM for the transcription factor from JASPAR database
    log += "## Step 1: Retrieving transcription factor PWM\n"

    try:
        # Search for the TF in JASPAR database
        jaspar_url = f"https://jaspar.genereg.net/api/v1/matrix/?name={tf_name}"
        response = requests.get(jaspar_url)
        tf_data = response.json()

        if not tf_data["results"] or len(tf_data["results"]) == 0:
            log += f"No PWM found for {tf_name} in JASPAR database.\n"
            return log

        # Get the first match's ID
        matrix_id = tf_data["results"][0]["matrix_id"]
        log += f"Found PWM with ID: {matrix_id}\n"

        # Retrieve the PWM
        pwm_url = f"https://jaspar.genereg.net/api/v1/matrix/{matrix_id}.pfm"
        pwm_response = requests.get(pwm_url)

        # Parse the PWM
        handle = io.StringIO(pwm_response.text)
        motif = motifs.read(handle, "jaspar")
        log += f"Successfully retrieved PWM for {tf_name}\n"

        # Calculate position-specific scoring matrix (PSSM)
        pssm = motif.pssm

        # Step 2: Scan the sequence for binding sites
        log += "\n## Step 2: Scanning sequence for binding sites\n"
        log += f"Sequence length: {len(sequence)} bp\n"
        log += f"Using score threshold: {threshold}\n\n"

        # Find binding sites
        binding_sites = []
        max_score = pssm.max
        min_score = pssm.min

        for position, score in pssm.search(Seq(sequence), threshold=threshold):
            relative_score = (score - min_score) / (max_score - min_score)
            if position >= 0:
                strand = "+"
                site_seq = sequence[position : position + len(pssm)]
            else:
                strand = "-"
                site_seq = sequence[len(sequence) + position - len(pssm) : len(sequence) + position]

            binding_sites.append(
                {
                    "position": abs(position),
                    "strand": strand,
                    "score": score,
                    "relative_score": relative_score,
                    "sequence": site_seq,
                }
            )

        # Step 3: Summarize results
        log += f"## Step 3: Results - Found {len(binding_sites)} potential binding sites\n\n"

        if binding_sites:
            # Sort by position
            binding_sites.sort(key=lambda x: x["position"])

            # Create a table of results
            log += "| Position | Strand | Sequence | Score | Relative Score |\n"
            log += "|----------|--------|----------|-------|---------------|\n"

            for site in binding_sites:
                log += f"| {site['position']} | {site['strand']} | {site['sequence']} | {site['score']:.2f} | {site['relative_score']:.2f} |\n"
        else:
            log += "No binding sites found meeting the threshold criteria.\n"

        # Save results to file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(f"# {tf_name} binding sites in sequence\n")
                f.write("Position\tStrand\tSequence\tScore\tRelative Score\n")

                for site in binding_sites:
                    f.write(
                        f"{site['position']}\t{site['strand']}\t{site['sequence']}\t{site['score']:.2f}\t{site['relative_score']:.2f}\n"
                    )

            log += f"\nResults saved to file: {output_file}\n"

    except Exception as e:
        log += f"\n## Error occurred during analysis: {str(e)}\n"

    log += "\n## Analysis complete\n"
    return log


def fit_genomic_prediction_model(
    genotypes,
    phenotypes,
    fixed_effects=None,
    model_type="additive",
    output_file="genomic_prediction_results.csv",
):
    """Fit a linear mixed model for genomic prediction using genotype and phenotype data.

    Parameters
    ----------
    genotypes : numpy.ndarray
        Matrix of genotype data, with individuals in rows and markers in columns.
        Values are typically coded as 0, 1, 2 for additive models or with specific
        encoding for dominance effects.
    phenotypes : numpy.ndarray
        Vector or matrix of phenotype data, with individuals in rows and traits in columns.
    fixed_effects : numpy.ndarray, optional
        Matrix of fixed effects (e.g., environment, management), with individuals in rows
        and effects in columns.
    model_type : str, optional
        Type of genetic model to fit: "additive" or "additive_dominance".
    output_file : str, optional
        File name to save the results.

    Returns
    -------
    str
        Research log summarizing the genomic prediction analysis, including model parameters,
        variance components, breeding values, and prediction accuracy metrics.

    """
    import pandas as pd
    from scipy import linalg

    # Initialize research log
    log = "# Multi-trait Genomic Prediction Analysis\n\n"
    log += f"Model type: {model_type}\n"

    # Basic validation
    n_individuals, n_markers = genotypes.shape
    n_pheno, n_traits = (phenotypes.shape[0], 1) if phenotypes.ndim == 1 else phenotypes.shape

    if n_individuals != n_pheno:
        raise ValueError(f"Number of individuals in genotypes ({n_individuals}) and phenotypes ({n_pheno}) don't match")

    log += f"Number of individuals: {n_individuals}\n"
    log += f"Number of markers: {n_markers}\n"
    log += f"Number of traits: {n_traits}\n\n"

    # Ensure phenotypes is 2D
    if phenotypes.ndim == 1:
        phenotypes = phenotypes.reshape(-1, 1)

    # Center genotypes (common preprocessing step)
    genotypes_centered = genotypes - np.mean(genotypes, axis=0)

    # Create genomic relationship matrix (G)
    if model_type == "additive":
        # Additive genomic relationship matrix
        G = np.dot(genotypes_centered, genotypes_centered.T) / n_markers
        log += "Constructed additive genomic relationship matrix (G)\n\n"
    elif model_type == "additive_dominance":
        # For additive-dominance model, we need both A (additive) and D (dominance) matrices
        # Assuming genotypes are coded as {0,1,2} for {aa,Aa,AA}
        # Create dominance matrix - simple implementation assuming standard coding
        dom_genotypes = np.zeros_like(genotypes)
        # Code heterozygotes (1) as 1, homozygotes (0,2) as 0 for dominance effects
        dom_genotypes[genotypes == 1] = 1
        dom_genotypes_centered = dom_genotypes - np.mean(dom_genotypes, axis=0)

        # Additive and dominance matrices
        G_a = np.dot(genotypes_centered, genotypes_centered.T) / n_markers
        G_d = np.dot(dom_genotypes_centered, dom_genotypes_centered.T) / n_markers
        log += "Constructed additive (G_a) and dominance (G_d) genomic relationship matrices\n\n"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Initialize results storage
    trait_results = []

    # Fit model for each trait
    for trait_idx in range(n_traits):
        trait_phenotypes = phenotypes[:, trait_idx]
        log += f"## Trait {trait_idx + 1} Analysis\n\n"

        # Handle fixed effects if provided
        if fixed_effects is not None:
            # Simple fixed effects adjustment - more complex models would use proper mixed model fitting
            X = fixed_effects
            # Fit fixed effects model
            beta = np.linalg.lstsq(X, trait_phenotypes, rcond=None)[0]
            # Adjust phenotypes for fixed effects
            y_adj = trait_phenotypes - X @ beta
            log += f"Applied adjustment for {X.shape[1]} fixed effects\n"
        else:
            y_adj = trait_phenotypes
            log += "No fixed effects provided\n"

        # Fit mixed model
        if model_type == "additive":
            # Simplified REML estimation for variance components
            # In practice, specialized libraries like pyGWAS, GCTA or R's ASReml would be used

            # Initial variance component estimates
            var_g_init = np.var(y_adj) * 0.5  # genetic variance
            var_e_init = np.var(y_adj) * 0.5  # residual variance

            # Simple EM-like algorithm for variance component estimation
            # (In practice, use dedicated software for proper REML)
            for _ in range(5):  # Few iterations for demonstration
                # Construct mixed model equations
                V = var_g_init * G + var_e_init * np.eye(n_individuals)
                V_inv = linalg.inv(V)

                # Update variance components
                P = V_inv - V_inv @ np.ones((n_individuals, 1)) @ np.ones((1, n_individuals)) @ V_inv / (
                    np.ones((1, n_individuals)) @ V_inv @ np.ones((n_individuals, 1))
                )
                var_g_new = (y_adj.T @ P @ G @ P @ y_adj) / np.trace(P @ G)
                var_e_new = (y_adj.T @ P @ P @ y_adj) / np.trace(P)

                # Update estimates
                var_g_init = max(0.01, var_g_new)
                var_e_init = max(0.01, var_e_new)

            # Final variance components
            var_g = var_g_init
            var_e = var_e_init

            # Calculate heritability
            heritability = var_g / (var_g + var_e)

            # BLUP solutions for breeding values
            V = var_g * G + var_e * np.eye(n_individuals)
            V_inv = linalg.inv(V)
            breeding_values = var_g * G @ V_inv @ y_adj

            # Predicted phenotypes
            predicted_phenotypes = breeding_values

            # Calculate accuracy
            accuracy = np.corrcoef(trait_phenotypes, predicted_phenotypes)[0, 1]

            # Log results
            log += f"Estimated additive genetic variance: {var_g:.4f}\n"
            log += f"Estimated residual variance: {var_e:.4f}\n"
            log += f"Estimated heritability: {heritability:.4f}\n"
            log += f"Prediction accuracy (correlation): {accuracy:.4f}\n\n"

            # Store results
            trait_result = {
                "trait": trait_idx + 1,
                "var_g": var_g,
                "var_e": var_e,
                "heritability": heritability,
                "accuracy": accuracy,
                "breeding_values": breeding_values,
                "predicted_phenotypes": predicted_phenotypes,
            }
            trait_results.append(trait_result)

        elif model_type == "additive_dominance":
            # Similar approach but with both additive and dominance effects
            # Initial variance component estimates
            var_a_init = np.var(y_adj) * 0.4  # additive variance
            var_d_init = np.var(y_adj) * 0.1  # dominance variance
            var_e_init = np.var(y_adj) * 0.5  # residual variance

            # Simple estimation iterations
            for _ in range(5):  # Few iterations for demonstration
                # Construct mixed model equations
                V = var_a_init * G_a + var_d_init * G_d + var_e_init * np.eye(n_individuals)
                V_inv = linalg.inv(V)

                # Update variance components (simplified)
                P = V_inv - V_inv @ np.ones((n_individuals, 1)) @ np.ones((1, n_individuals)) @ V_inv / (
                    np.ones((1, n_individuals)) @ V_inv @ np.ones((n_individuals, 1))
                )
                var_a_new = (y_adj.T @ P @ G_a @ P @ y_adj) / np.trace(P @ G_a)
                var_d_new = (y_adj.T @ P @ G_d @ P @ y_adj) / np.trace(P @ G_d)
                var_e_new = (y_adj.T @ P @ P @ y_adj) / np.trace(P)

                # Update estimates
                var_a_init = max(0.01, var_a_new)
                var_d_init = max(0.01, var_d_new)
                var_e_init = max(0.01, var_e_new)

            # Final variance components
            var_a = var_a_init
            var_d = var_d_init
            var_e = var_e_init

            # Calculate heritabilities
            narrow_heritability = var_a / (var_a + var_d + var_e)
            broad_heritability = (var_a + var_d) / (var_a + var_d + var_e)

            # BLUP solutions for breeding values and dominance deviations
            V = var_a * G_a + var_d * G_d + var_e * np.eye(n_individuals)
            V_inv = linalg.inv(V)
            breeding_values = var_a * G_a @ V_inv @ y_adj
            dominance_deviations = var_d * G_d @ V_inv @ y_adj

            # Predicted phenotypes
            predicted_phenotypes = breeding_values + dominance_deviations

            # Calculate accuracy
            accuracy = np.corrcoef(trait_phenotypes, predicted_phenotypes)[0, 1]

            # Log results
            log += f"Estimated additive genetic variance: {var_a:.4f}\n"
            log += f"Estimated dominance genetic variance: {var_d:.4f}\n"
            log += f"Estimated residual variance: {var_e:.4f}\n"
            log += f"Estimated narrow-sense heritability: {narrow_heritability:.4f}\n"
            log += f"Estimated broad-sense heritability: {broad_heritability:.4f}\n"
            log += f"Prediction accuracy (correlation): {accuracy:.4f}\n\n"

            # Store results
            trait_result = {
                "trait": trait_idx + 1,
                "var_a": var_a,
                "var_d": var_d,
                "var_e": var_e,
                "narrow_heritability": narrow_heritability,
                "broad_heritability": broad_heritability,
                "accuracy": accuracy,
                "breeding_values": breeding_values,
                "dominance_deviations": dominance_deviations,
                "predicted_phenotypes": predicted_phenotypes,
            }
            trait_results.append(trait_result)

    # Save results to file
    results_df = pd.DataFrame()

    for i, trait_result in enumerate(trait_results):
        # Create individual-level results
        ind_data = {
            "individual": np.arange(1, n_individuals + 1),
            f"trait_{i + 1}_observed": phenotypes[:, i],
            f"trait_{i + 1}_predicted": trait_result["predicted_phenotypes"],
            f"trait_{i + 1}_breeding_value": trait_result["breeding_values"],
        }

        if model_type == "additive_dominance":
            ind_data[f"trait_{i + 1}_dominance_deviation"] = trait_result["dominance_deviations"]

        # Create or append to dataframe
        if i == 0:
            results_df = pd.DataFrame(ind_data)
        else:
            for key, value in ind_data.items():
                if key != "individual":  # Skip duplicating individual column
                    results_df[key] = value

    # Save to CSV
    results_df.to_csv(output_file, index=False)
    log += f"Results saved to {output_file}\n"

    return log


def perform_pcr_and_gel_electrophoresis(
    genomic_dna,
    forward_primer=None,
    reverse_primer=None,
    target_region=None,
    annealing_temp=58,
    extension_time=30,
    cycles=35,
    gel_percentage=2.0,
    output_prefix="pcr_result",
):
    """Performs PCR amplification of a target transgene and visualizes results using agarose gel electrophoresis.

    Parameters
    ----------
    genomic_dna : str
        Path to file containing genomic DNA sequence in FASTA format or the sequence itself
    forward_primer : str, optional
        Forward primer sequence. If not provided, will be designed based on target_region
    reverse_primer : str, optional
        Reverse primer sequence. If not provided, will be designed based on target_region
    target_region : tuple, optional
        Tuple of (start, end) positions for the target region in the genomic DNA
    annealing_temp : float, default=58
        Annealing temperature for PCR in °C
    extension_time : int, default=30
        Extension time in seconds
    cycles : int, default=35
        Number of PCR cycles
    gel_percentage : float, default=2.0
        Percentage of agarose gel
    output_prefix : str, default="pcr_result"
        Prefix for output files

    Returns
    -------
    str
        Research log summarizing the PCR and gel electrophoresis procedures and results

    """
    import datetime

    import matplotlib.pyplot as plt
    import numpy as np
    from Bio import SeqIO
    from Bio.Seq import Seq

    log = f"PCR AMPLIFICATION AND GEL ELECTROPHORESIS LOG - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    log += "=" * 80 + "\n\n"

    # Step 1: Load the genomic DNA
    log += "STEP 1: PREPARING GENOMIC DNA\n"
    if os.path.isfile(genomic_dna):
        try:
            record = SeqIO.read(genomic_dna, "fasta")
            dna_sequence = str(record.seq)
            log += f"- Loaded genomic DNA from file: {genomic_dna}\n"
            log += f"- Sequence length: {len(dna_sequence)} bp\n"
        except Exception as e:
            log += f"- Error loading DNA file: {str(e)}\n"
            return log
    else:
        dna_sequence = genomic_dna
        log += "- Using provided DNA sequence\n"
        log += f"- Sequence length: {len(dna_sequence)} bp\n"

    log += "\n"

    # Step 2: Design or validate primers
    log += "STEP 2: PCR PRIMER PREPARATION\n"

    if forward_primer is None or reverse_primer is None:
        if target_region is None:
            log += "- Error: Either primers or target region must be provided\n"
            return log

        # Simple primer design based on target region
        start, end = target_region

        if forward_primer is None:
            # Take 20bp from the start of the target region for forward primer
            forward_primer = dna_sequence[start : start + 20]
            log += "- Designed forward primer based on target region\n"

        if reverse_primer is None:
            # Take 20bp from the end of the target region for reverse primer (reverse complement)
            reverse_seq = Seq(dna_sequence[end - 20 : end])
            reverse_primer = str(reverse_seq.reverse_complement())
            log += "- Designed reverse primer based on target region\n"

    log += f"- Forward primer: 5'-{forward_primer}-3' ({len(forward_primer)} bp)\n"
    log += f"- Reverse primer: 5'-{reverse_primer}-3' ({len(reverse_primer)} bp)\n"

    # Step 3: PCR Setup and Amplification
    log += "\nSTEP 3: PCR AMPLIFICATION\n"
    log += "- PCR reaction setup:\n"
    log += "  * Template DNA: Genomic DNA\n"
    log += f"  * Forward primer: 5'-{forward_primer}-3'\n"
    log += f"  * Reverse primer: 5'-{reverse_primer}-3'\n"
    log += f"  * Annealing temperature: {annealing_temp}°C\n"
    log += f"  * Extension time: {extension_time} seconds\n"
    log += f"  * Number of cycles: {cycles}\n"

    # Simulate PCR by finding binding sites and determining amplicon size
    amplicon_size = None
    amplicon_sequence = None

    # Find forward primer binding site
    fwd_pos = dna_sequence.find(forward_primer)
    if fwd_pos == -1:
        log += "- Warning: Forward primer binding site not found in sequence\n"

    # Find reverse primer binding site (need to search for reverse complement)
    rev_primer_seq = Seq(reverse_primer)
    rev_primer_rc = str(rev_primer_seq.reverse_complement())
    rev_pos = dna_sequence.find(rev_primer_rc)
    if rev_pos == -1:
        log += "- Warning: Reverse primer binding site not found in sequence\n"

    # If we found both binding sites, calculate amplicon size
    if fwd_pos != -1 and rev_pos != -1:
        if fwd_pos < rev_pos:
            amplicon_size = rev_pos + len(reverse_primer) - fwd_pos
            amplicon_sequence = dna_sequence[fwd_pos : rev_pos + len(reverse_primer)]
            log += "- PCR amplification successful\n"
            log += f"- Amplicon size: {amplicon_size} bp\n"
        else:
            log += "- Error: Primer binding sites are in incorrect orientation\n"
    # Simulate based on target region if provided
    elif target_region is not None:
        start, end = target_region
        amplicon_size = end - start + len(forward_primer) + len(reverse_primer)
        log += "- PCR amplification simulated based on target region\n"
        log += f"- Expected amplicon size: {amplicon_size} bp\n"
    else:
        log += "- PCR amplification failed: could not determine amplicon size\n"
        return log

    # Step 4: Gel Electrophoresis
    log += "\nSTEP 4: AGAROSE GEL ELECTROPHORESIS\n"
    log += f"- Prepared {gel_percentage}% agarose gel\n"
    log += "- Loaded PCR product alongside DNA ladder\n"
    log += "- Ran electrophoresis at 100V for 45 minutes\n"

    # Create a simulated gel image
    fig, ax = plt.subplots(figsize=(6, 8))

    # Draw gel lanes
    ax.add_patch(plt.Rectangle((0, 0), 6, 10, color="lightgray", alpha=0.5))

    # DNA Ladder (100bp increments)
    ladder_sizes = [100, 200, 300, 500, 700, 1000, 1500, 2000]
    ladder_positions = [10 - (np.log(size) / np.log(2000) * 8) for size in ladder_sizes]

    # Plot ladder
    for pos, size in zip(ladder_positions, ladder_sizes, strict=False):
        ax.add_patch(plt.Rectangle((0.5, pos - 0.1), 1, 0.2, color="black", alpha=0.8))
        ax.text(0.2, pos, f"{size}bp", fontsize=8, ha="right", va="center")

    # Plot sample band
    if amplicon_size:
        sample_position = 10 - (np.log(amplicon_size) / np.log(2000) * 8)
        ax.add_patch(plt.Rectangle((3.5, sample_position - 0.15), 1, 0.3, color="black", alpha=0.8))
        ax.text(
            4.5,
            sample_position,
            f"{amplicon_size}bp",
            fontsize=8,
            ha="left",
            va="center",
        )

    # Set up the plot
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 10)
    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(["Ladder", "Sample"])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"{gel_percentage}% Agarose Gel")

    # Save the gel image
    gel_image_path = f"{output_prefix}_gel.png"
    plt.savefig(gel_image_path, dpi=300, bbox_inches="tight")
    plt.close()

    log += f"- Gel image saved as: {gel_image_path}\n"

    # Results interpretation
    log += "\nRESULTS INTERPRETATION:\n"
    if amplicon_size:
        log += f"- Detected band at approximately {amplicon_size} bp\n"

        # Save amplicon sequence if available
        if amplicon_sequence:
            seq_file = f"{output_prefix}_amplicon.fasta"
            with open(seq_file, "w") as f:
                f.write(f">PCR_Amplicon_{amplicon_size}bp\n")
                f.write(amplicon_sequence)
            log += f"- Amplicon sequence saved as: {seq_file}\n"
    else:
        log += "- No bands detected\n"

    return log


def analyze_protein_phylogeny(
    fasta_sequences,
    output_dir="./",
    alignment_method="clustalw",
    tree_method="fasttree",
):
    """Perform phylogenetic analysis on a set of protein sequences.

    This function takes protein sequences in FASTA format, performs multiple sequence alignment,
    constructs a phylogenetic tree, and visualizes the evolutionary relationships.

    Parameters
    ----------
    fasta_sequences : str
        Path to a FASTA file containing protein sequences or a string with FASTA-formatted sequences
    output_dir : str, optional
        Directory to save output files (default: current directory)
    alignment_method : str, optional
        Method for sequence alignment: "clustalw", "muscle", or "pre-aligned" (default: "clustalw")
    tree_method : str, optional
        Method for tree construction: "iqtree" (default: "iqtree")

    Returns
    -------
    str
        Research log summarizing the phylogenetic analysis process

    """
    import datetime
    import subprocess
    import tempfile

    from Bio import AlignIO, Phylo, SeqIO
    from Bio.Align.Applications import ClustalwCommandline, MuscleCommandline

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize log
    log = []
    log.append(f"Phylogenetic Analysis - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append("=" * 50)

    # Check if input is a file path or string content
    if os.path.isfile(fasta_sequences):
        input_file = fasta_sequences
        log.append(f"Using sequences from file: {input_file}")
    else:
        # Create temporary file with the string content
        temp_fasta = tempfile.NamedTemporaryFile(delete=False, suffix=".fasta", dir=output_dir)
        temp_fasta.write(fasta_sequences.encode())
        temp_fasta.close()
        input_file = temp_fasta.name
        log.append(f"Created temporary FASTA file from provided sequences: {input_file}")

    # Count sequences
    try:
        sequences = list(SeqIO.parse(input_file, "fasta"))
        log.append(f"Loaded {len(sequences)} protein sequences")
    except Exception as e:
        log.append(f"Warning: Could not parse sequences as FASTA: {str(e)}")
        # This might be pre-aligned data already
        sequences = []

    # Create filenames for outputs
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    alignment_file = os.path.join(output_dir, f"{base_name}_aligned.aln")
    tree_file = os.path.join(output_dir, f"{base_name}_tree.nwk")
    tree_image = os.path.join(output_dir, f"{base_name}_phylogeny.png")

    # Perform multiple sequence alignment
    log.append("\nStep 1: Multiple Sequence Alignment")

    # Special case for pre-aligned input
    if alignment_method.lower() == "pre-aligned":
        log.append("Using pre-aligned sequences")
        try:
            # If the input is already an alignment file, copy it to the alignment_file path
            with open(input_file) as src, open(alignment_file, "w") as dst:
                dst.write(src.read())
            log.append(f"Copied pre-aligned file to: {alignment_file}")
        except Exception as e:
            log.append(f"Error processing pre-aligned sequences: {str(e)}")
            return "\n".join(log)
    elif alignment_method.lower() == "clustalw":
        log.append("Using Clustal Omega for alignment")
        try:
            clustalw_cline = ClustalwCommandline("clustalw", infile=input_file, outfile=alignment_file)
            stdout, stderr = clustalw_cline()
            log.append("Alignment completed successfully")
        except Exception as e:
            log.append(f"Error during alignment: {str(e)}")
            # Try alternative approach using MUSCLE if ClustalW fails
            alignment_method = "muscle"

    if alignment_method.lower() == "muscle":
        log.append("Using MUSCLE for alignment")
        try:
            muscle_cline = MuscleCommandline("muscle", input=input_file, out=alignment_file)
            stdout, stderr = muscle_cline()
            log.append("Alignment completed successfully")
        except Exception as e:
            log.append(f"Error during MUSCLE alignment: {str(e)}")
            log.append("Attempting to use Biopython's built-in pairwise2 alignment as fallback")

            from Bio import pairwise2

            # Create a simple progressive alignment
            ref_seq = sequences[0]
            alignments = []

            for seq in sequences:
                alignment = pairwise2.align.globalxx(ref_seq.seq, seq.seq)[0]
                alignments.append(alignment)

            # Write alignment to file in CLUSTAL format
            with open(alignment_file, "w") as f:
                f.write("CLUSTAL W (1.83) multiple sequence alignment\n\n")
                for i, seq in enumerate(sequences):
                    f.write(f"{seq.id.ljust(10)} {alignments[i][1]}\n")
                # Add consensus line with asterisks
                f.write(" " * 10 + " " * len(alignments[0][1]) + "\n")

            log.append("Created basic alignment using Biopython's pairwise2")

    # Verify alignment file exists
    if not os.path.exists(alignment_file):
        log.append(f"Error: Alignment file {alignment_file} does not exist")
        return "\n".join(log)

    # Build phylogenetic tree
    log.append("\nStep 2: Phylogenetic Tree Construction")

    if tree_method.lower() == "iqtree":
        log.append("Using IQ-TREE for phylogenetic tree construction")
        try:
            cmd = f"iqtree -s {alignment_file} -m LG -bb 1000 -pre {os.path.join(output_dir, base_name)}"
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                capture_output=True,
            )
            # IQ-TREE creates files with .treefile extension
            iqtree_file = os.path.join(output_dir, f"{base_name}.treefile")
            if os.path.exists(iqtree_file):
                # Rename to our standard name
                os.rename(iqtree_file, tree_file)
            log.append("Tree construction completed successfully")
        except Exception as e:
            log.append(f"Error during IQ-TREE execution: {str(e)}")
            log.append("Falling back to neighbor-joining method")

            try:
                from Bio.Phylo.TreeConstruction import (
                    DistanceCalculator,
                    DistanceTreeConstructor,
                )

                # Try to read the alignment in different formats
                alignment = None
                for format in ["clustal", "fasta"]:
                    try:
                        alignment = AlignIO.read(alignment_file, format)
                        break
                    except Exception:
                        continue

                if alignment is None:
                    log.append("Could not parse alignment file in any supported format")
                    # Create a simple text-based tree file as a fallback
                    with open(tree_file, "w") as f:
                        f.write("(protein1:0.1,protein2:0.2,(protein3:0.3,protein4:0.4):0.5);")
                    log.append("Created placeholder tree file")
                else:
                    # Calculate the distance matrix
                    calculator = DistanceCalculator("identity")
                    dm = calculator.get_distance(alignment)

                    # Construct the tree
                    constructor = DistanceTreeConstructor()
                    tree = constructor.nj(dm)

                    # Write the tree to file
                    Phylo.write(tree, tree_file, "newick")
                    log.append("Created tree using neighbor-joining method")
            except Exception as e:
                log.append(f"Error during fallback tree construction: {str(e)}")
                # Create a simple text-based tree file as a final fallback
                with open(tree_file, "w") as f:
                    f.write("(protein1:0.1,protein2:0.2,(protein3:0.3,protein4:0.4):0.5);")
                log.append("Created placeholder tree file as final fallback")
    else:
        log.append(f"Unsupported tree method: {tree_method}")
        return "\n".join(log)

    # Verify tree file exists
    if not os.path.exists(tree_file):
        log.append(f"Error: Tree file {tree_file} does not exist")
        return "\n".join(log)

    # Visualize the tree
    log.append("\nStep 3: Phylogenetic Tree Visualization")
    try:
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
        import matplotlib.pyplot as plt

        tree = Phylo.read(tree_file, "newick")
        fig = plt.figure(figsize=(10, len(sequences) * 0.3 if sequences else 5))
        axes = fig.add_subplot(1, 1, 1)
        Phylo.draw(tree, axes=axes, do_show=False)
        plt.savefig(tree_image, dpi=300, bbox_inches="tight")
        plt.close()
        log.append(f"Tree visualization saved to: {tree_image}")
    except Exception as e:
        log.append(f"Error during tree visualization: {str(e)}")

    # Summary
    log.append("\nSummary:")
    log.append(f"- Input sequences: {len(sequences) if sequences else 'pre-aligned data'}")
    log.append(f"- Alignment file: {alignment_file}")
    log.append(f"- Phylogenetic tree file: {tree_file}")
    log.append(f"- Tree visualization: {tree_image}")

    return "\n".join(log)
