"""
Gene Regulation Validation Tools for Biomni
=============================================

Validates computational gene regulatory predictions against experimental
perturbation data from the Replogle Perturb-seq atlas (~11,000 gene knockdowns
in K562 cells, ~2,000 in RPE1 cells).

Based on TruthSeq (https://github.com/rsflinn/truthseq).

Two tools:
  1. validate_gene_regulatory_claims - Grade each claim as VALIDATED,
     PARTIALLY_SUPPORTED, WEAK, CONTRADICTED, or UNTESTABLE.
  2. test_regulatory_specificity - Permutation test: are your upstream
     regulators special, or would random genes score equally well?

Data requirement: Replogle Perturb-seq knockdown effects parquet file(s).
  - K562: replogle_knockdown_effects_K562.parquet
  - RPE1: replogle_knockdown_effects_RPE1.parquet (optional)
  - Stats: replogle_knockdown_stats.parquet (optional, improves grading)
  Expected in {data_lake_path}/truthseq/

Install: biomni_env/new_software_truthseq.sh
"""

import logging
import os

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
RANDOM_SEED = 42
NULL_SAMPLE_SIZE = 500
PERCENTILE_VALIDATED = 90
PERCENTILE_PARTIAL = 50


# ── Internal helpers (not exposed as tools) ────────────────────────────────


def _load_replogle(data_lake_path: str):
    """Load Replogle perturbation data from the Biomni datalake."""
    truthseq_dir = os.path.join(data_lake_path, "truthseq")
    dfs = []
    for fname, cell_line in [
        ("replogle_knockdown_effects_K562.parquet", "K562"),
        ("replogle_knockdown_effects_RPE1.parquet", "RPE1"),
        # Also accept the original TruthSeq filename
        ("replogle_knockdown_effects.parquet", "K562"),
    ]:
        path = os.path.join(truthseq_dir, fname)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if "cell_line" not in df.columns:
                df["cell_line"] = cell_line
            dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def _load_stats(data_lake_path: str):
    """Load per-knockdown distribution statistics."""
    truthseq_dir = os.path.join(data_lake_path, "truthseq")
    for fname in [
        "replogle_knockdown_stats.parquet",
        "replogle_knockdown_stats_K562.parquet",
    ]:
        path = os.path.join(truthseq_dir, fname)
        if os.path.exists(path):
            return pd.read_parquet(path)
    return None


def _parse_claims(claims_input):
    """
    Parse claims from either a CSV file path or an inline list of dicts.

    Each claim needs: upstream_gene, downstream_gene, predicted_direction (UP/DOWN).
    """
    if isinstance(claims_input, str) and os.path.exists(claims_input):
        df = pd.read_csv(claims_input)
    elif isinstance(claims_input, list):
        df = pd.DataFrame(claims_input)
    elif isinstance(claims_input, pd.DataFrame):
        df = claims_input
    else:
        raise ValueError(f"claims_input must be a CSV path, list of dicts, or DataFrame. Got: {type(claims_input)}")

    required = ["upstream_gene", "downstream_gene", "predicted_direction"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Claims missing required columns: {missing}. "
            f"Found: {list(df.columns)}. "
            f"Each claim needs: upstream_gene, downstream_gene, predicted_direction (UP or DOWN)."
        )

    df["predicted_direction"] = df["predicted_direction"].str.upper().str.strip()
    return df


def _compute_percentile_from_stats(z_score, stats_row):
    """Compute approximate percentile using stored quantile breakpoints."""
    abs_z = abs(z_score)
    quantile_points = []
    quantile_values = []
    for col in sorted(stats_row.index):
        if col.startswith("q") and col[1:].isdigit():
            pct = int(col[1:])
            quantile_points.append(pct)
            quantile_values.append(float(stats_row[col]))
    if not quantile_points:
        return None
    qp = np.array(quantile_points, dtype=float)
    qv = np.array(quantile_values, dtype=float)
    valid = ~np.isnan(qv)
    if not valid.any():
        return None
    qp, qv = qp[valid], qv[valid]
    if abs_z <= qv[0]:
        return float(qp[0])
    if abs_z >= qv[-1]:
        return min(99.9, float(qp[-1]) + 0.5)
    percentile = float(np.interp(abs_z, qv, qp))
    return None if np.isnan(percentile) else percentile


def _grade_claim(percentile, direction_match, perturb_status, de_sig=False):
    """Assign a confidence grade to a single claim."""
    if perturb_status == "DATA_FOUND":
        if direction_match is False:
            return "CONTRADICTED", "Effect direction opposes prediction."
        elif percentile >= PERCENTILE_VALIDATED and direction_match and de_sig:
            return "VALIDATED", (f"Top {100 - percentile:.0f}% effect, direction matches, disease tissue confirms.")
        elif percentile >= PERCENTILE_VALIDATED and direction_match:
            return "PARTIALLY_SUPPORTED", (
                f"Top {100 - percentile:.0f}% effect, direction matches. No disease tissue confirmation."
            )
        elif percentile >= PERCENTILE_PARTIAL and direction_match:
            return "PARTIALLY_SUPPORTED", (f"Moderate effect ({percentile:.0f}th percentile), direction matches.")
        else:
            return "WEAK", (f"Effect not stronger than random ({percentile:.0f}th percentile).")
    elif perturb_status == "BELOW_THRESHOLD":
        if de_sig:
            return "PARTIALLY_SUPPORTED", ("Below |Z|>1 in perturbation, but disease tissue shows dysregulation.")
        return "WEAK", "Below |Z|>1 in perturbation, no disease tissue support."
    elif perturb_status == "UPSTREAM_NOT_TESTED":
        if de_sig:
            return "PARTIALLY_SUPPORTED", ("No perturbation data, but disease tissue shows dysregulation.")
        return "UNTESTABLE", "Upstream gene not in knockdown dataset."
    return "UNTESTABLE", f"Insufficient data ({perturb_status})."


# ══════════════════════════════════════════════════════════════════════════
# TOOL 1: Validate gene regulatory claims
# ══════════════════════════════════════════════════════════════════════════


def validate_gene_regulatory_claims(
    claims_input: str,
    data_lake_path: str,
    output_folder: str = "./tmp/",
    cell_type: str = "all",
) -> str:
    """
    Validate gene regulatory predictions against experimental knockdown data.

    Takes a CSV of claims (upstream_gene, downstream_gene, predicted_direction)
    and checks each against the Replogle Perturb-seq atlas. Each claim is graded:
      - VALIDATED: Strong perturbation effect + correct direction + disease tissue support
      - PARTIALLY_SUPPORTED: Some evidence, missing one tier
      - WEAK: Effect not stronger than random genes
      - CONTRADICTED: Significant effect in the OPPOSITE direction
      - UNTESTABLE: Upstream gene not in the knockdown dataset

    Args:
        claims_input: Path to a CSV file with columns: upstream_gene,
            downstream_gene, predicted_direction (UP or DOWN).
        data_lake_path: Path to the Biomni datalake root.
        output_folder: Directory for output files.
        cell_type: Which cell line to use: "K562", "RPE1", or "all" (default).

    Returns:
        Step-by-step progress log with results summary.
    """
    steps = []
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Load claims
    steps.append("Step 1: Loading gene regulatory claims...")
    try:
        claims = _parse_claims(claims_input)
    except ValueError as e:
        return f"ERROR: {e}"
    steps.append(f"  Loaded {len(claims)} claims from {claims_input}")

    # Step 2: Load perturbation data
    steps.append("Step 2: Loading Replogle Perturb-seq knockdown data...")
    replogle_df = _load_replogle(data_lake_path)
    if replogle_df is None:
        return "\n".join(
            steps
            + [
                "  ERROR: No Replogle data found. Expected parquet files in "
                f"{data_lake_path}/truthseq/. Run the install script first: "
                "biomni_env/new_software_truthseq.sh"
            ]
        )

    # Filter by cell type if requested
    if cell_type != "all" and "cell_line" in replogle_df.columns:
        replogle_df = replogle_df[replogle_df["cell_line"] == cell_type]
    cell_types = replogle_df["cell_line"].unique() if "cell_line" in replogle_df.columns else ["K562"]
    steps.append(
        f"  Loaded {len(replogle_df):,} knockdown-effect pairs "
        f"across {len(cell_types)} cell type(s): {', '.join(cell_types)}"
    )

    stats_df = _load_stats(data_lake_path)
    if stats_df is not None:
        steps.append(f"  Loaded distribution stats for {len(stats_df)} knockdowns")

    # Step 3: Validate each claim
    steps.append("Step 3: Validating each claim against perturbation data...")
    np.random.seed(RANDOM_SEED)
    available_kd = set(replogle_df["knocked_down_gene"].unique())
    stats_kd = set(stats_df["knocked_down_gene"].unique()) if stats_df is not None else set()
    all_kd = available_kd | stats_kd

    results = []
    for _idx, row in claims.iterrows():
        upstream = row["upstream_gene"]
        downstream = row["downstream_gene"]
        predicted_dir = row["predicted_direction"]

        if upstream not in all_kd:
            grade, reason = _grade_claim(0, None, "UPSTREAM_NOT_TESTED")
            results.append(
                {
                    "upstream_gene": upstream,
                    "downstream_gene": downstream,
                    "predicted_direction": predicted_dir,
                    "confidence_grade": grade,
                    "z_score": None,
                    "percentile": None,
                    "direction_match": None,
                    "cell_line": None,
                    "reason": reason,
                }
            )
            continue

        # Look up the effect
        kd_data = replogle_df[replogle_df["knocked_down_gene"] == upstream]
        target_hit = kd_data[kd_data["affected_gene"] == downstream]

        if len(target_hit) > 0:
            # Direct hit -- find best across cell types
            best_z = None
            best_pct = 0
            best_ct = None
            best_dir_match = None

            for ct in target_hit["cell_line"].unique() if "cell_line" in target_hit.columns else ["K562"]:
                ct_hits = target_hit[target_hit["cell_line"] == ct] if "cell_line" in target_hit.columns else target_hit
                z = float(ct_hits.iloc[0]["z_score"])
                obs_dir = "UP" if z < 0 else "DOWN"
                dir_match = obs_dir == predicted_dir

                # Percentile calculation
                pct = None
                if stats_df is not None:
                    ct_stats = stats_df[stats_df["knocked_down_gene"] == upstream]
                    if "cell_line" in stats_df.columns:
                        ct_stats = ct_stats[ct_stats["cell_line"] == ct]
                    if len(ct_stats) > 0:
                        pct = _compute_percentile_from_stats(z, ct_stats.iloc[0])
                if pct is None:
                    ct_kd = kd_data[kd_data["cell_line"] == ct] if "cell_line" in kd_data.columns else kd_data
                    all_z = ct_kd["z_score"].values
                    sample = np.random.choice(all_z, min(NULL_SAMPLE_SIZE, len(all_z)), replace=False)
                    pct = float(sp_stats.percentileofscore(np.abs(sample), abs(z)))

                if pct > best_pct:
                    best_z, best_pct, best_ct, best_dir_match = z, pct, ct, dir_match

            grade, reason = _grade_claim(best_pct, best_dir_match, "DATA_FOUND")
            results.append(
                {
                    "upstream_gene": upstream,
                    "downstream_gene": downstream,
                    "predicted_direction": predicted_dir,
                    "confidence_grade": grade,
                    "z_score": round(best_z, 4),
                    "percentile": round(best_pct, 1),
                    "direction_match": best_dir_match,
                    "cell_line": best_ct,
                    "reason": reason,
                }
            )

        elif stats_df is not None and upstream in stats_kd:
            # Below threshold
            grade, reason = _grade_claim(0, None, "BELOW_THRESHOLD")
            results.append(
                {
                    "upstream_gene": upstream,
                    "downstream_gene": downstream,
                    "predicted_direction": predicted_dir,
                    "confidence_grade": grade,
                    "z_score": None,
                    "percentile": None,
                    "direction_match": None,
                    "cell_line": "K562",
                    "reason": reason,
                }
            )
        else:
            grade, reason = _grade_claim(0, None, "BELOW_THRESHOLD")
            results.append(
                {
                    "upstream_gene": upstream,
                    "downstream_gene": downstream,
                    "predicted_direction": predicted_dir,
                    "confidence_grade": grade,
                    "z_score": None,
                    "percentile": None,
                    "direction_match": None,
                    "cell_line": None,
                    "reason": reason,
                }
            )

    results_df = pd.DataFrame(results)

    # Step 4: Summarize
    steps.append("Step 4: Grading complete. Summary:")
    grade_counts = results_df["confidence_grade"].value_counts()
    for grade_name in ["VALIDATED", "PARTIALLY_SUPPORTED", "WEAK", "CONTRADICTED", "UNTESTABLE"]:
        c = grade_counts.get(grade_name, 0)
        pct = c / len(results_df) * 100 if len(results_df) > 0 else 0
        steps.append(f"  {grade_name}: {c} ({pct:.1f}%)")

    # Step 5: Save output
    out_path = os.path.join(output_folder, "truthseq_validation_results.csv")
    results_df.to_csv(out_path, index=False)
    steps.append(f"Step 5: Results saved to {out_path}")

    # Per-claim detail
    steps.append("")
    steps.append("Per-claim results:")
    for _, r in results_df.iterrows():
        z_str = f"Z={r['z_score']:.2f}" if r["z_score"] is not None else "no data"
        pct_str = f"{r['percentile']:.0f}th pctl" if r["percentile"] is not None else ""
        dir_str = "matches" if r["direction_match"] else ("OPPOSES" if r["direction_match"] is False else "N/A")
        steps.append(
            f"  {r['upstream_gene']} -> {r['downstream_gene']}: "
            f"{r['confidence_grade']} ({z_str}, {dir_str}, {pct_str}) -- {r['reason']}"
        )

    return "\n".join(steps)


# ══════════════════════════════════════════════════════════════════════════
# TOOL 2: Test regulatory specificity
# ══════════════════════════════════════════════════════════════════════════


def test_regulatory_specificity(
    claims_input: str,
    data_lake_path: str,
    n_permutations: int = 1000,
    comparison_pool_file: str = None,
    output_folder: str = "./tmp/",
) -> str:
    """
    Permutation test: are your upstream regulators special for these downstream
    targets, or would random genes from the knockdown dataset score equally well?

    For each permutation, downstream targets stay fixed while upstream genes are
    replaced with random knockdown genes. If random regulators score as well as
    yours, the specific regulatory relationships are not special.

    This is the critical test that separates "these genes are individually
    important" from "these genes specifically regulate those targets."

    Args:
        claims_input: Path to a CSV file with columns: upstream_gene,
            downstream_gene, predicted_direction (UP or DOWN).
        data_lake_path: Path to the Biomni datalake root.
        n_permutations: Number of permutations to run (default 1000).
        comparison_pool_file: Optional path to a text file with one gene per
            line, restricting the random pool (e.g., other NDD genes). If not
            provided, all knockdown genes are used as the pool.
        output_folder: Directory for output files.

    Returns:
        Step-by-step progress log with specificity test results.
    """
    steps = []
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Load data
    steps.append("Step 1: Loading claims and perturbation data...")
    try:
        claims = _parse_claims(claims_input)
    except ValueError as e:
        return f"ERROR: {e}"
    steps.append(f"  Loaded {len(claims)} claims")

    replogle_df = _load_replogle(data_lake_path)
    if replogle_df is None:
        return "\n".join(
            steps + [f"  ERROR: No Replogle data found. Expected parquet files in {data_lake_path}/truthseq/"]
        )
    steps.append(f"  Loaded {len(replogle_df):,} knockdown-effect pairs")

    # Step 2: Set up comparison pool
    steps.append("Step 2: Setting up comparison pool...")
    all_kd_genes = list(set(replogle_df["knocked_down_gene"].unique()))

    if comparison_pool_file and os.path.exists(comparison_pool_file):
        with open(comparison_pool_file) as f:
            custom_genes = [line.strip() for line in f if line.strip()]
        pool_genes = [g for g in custom_genes if g in set(all_kd_genes)]
        if len(pool_genes) < 10:
            steps.append(
                f"  WARNING: Only {len(pool_genes)} of {len(custom_genes)} "
                f"custom pool genes are in knockdown data. Using all {len(all_kd_genes)} genes."
            )
            pool_genes = all_kd_genes
            pool_source = "all knockdowns (custom pool too small)"
        else:
            pool_source = f"custom gene list ({len(pool_genes)} genes)"
            steps.append(f"  Custom pool: {len(pool_genes)} genes in knockdown data")
    else:
        pool_genes = all_kd_genes
        pool_source = f"all knockdowns ({len(pool_genes)} genes)"

    steps.append(f"  Comparison pool: {pool_source}")

    # Step 3: Build lookup index
    steps.append("Step 3: Building knockdown effect index...")
    max_pairs = 2_000_000
    if len(replogle_df) > max_pairs:
        sim_df = replogle_df.sample(n=max_pairs, random_state=RANDOM_SEED)
    else:
        sim_df = replogle_df

    pair_z = dict(
        zip(
            zip(sim_df["knocked_down_gene"], sim_df["affected_gene"], strict=False),
            sim_df["z_score"],
            strict=False,
        )
    )

    # Per-knockdown |Z| distributions
    kd_abs_z = {}
    for kd_gene, group in sim_df.groupby("knocked_down_gene"):
        kd_abs_z[kd_gene] = np.sort(group["z_score"].abs().values)

    # Ensure user's claim pairs are indexed
    for _, row in claims.iterrows():
        up, down = row["upstream_gene"], row["downstream_gene"]
        if (up, down) not in pair_z:
            mask = (replogle_df["knocked_down_gene"] == up) & (replogle_df["affected_gene"] == down)
            hits = replogle_df.loc[mask]
            if len(hits) > 0:
                pair_z[(up, down)] = float(hits.iloc[0]["z_score"])
        if up not in kd_abs_z:
            kd_data = replogle_df[replogle_df["knocked_down_gene"] == up]
            if len(kd_data) > 0:
                kd_abs_z[up] = np.sort(kd_data["z_score"].abs().values[:5000])

    steps.append(f"  Index: {len(pair_z):,} pairs, {len(kd_abs_z)} knockdown distributions")

    # Step 4: Score function
    def score_set(upstreams, downstreams, directions):
        n_supported = 0
        percentiles = []
        n_dir_match = 0
        n_testable = 0
        for up, down, pred_dir in zip(upstreams, downstreams, directions, strict=False):
            if up not in set(pool_genes) and up not in kd_abs_z:
                continue
            n_testable += 1
            z = pair_z.get((up, down))
            if z is not None:
                obs_dir = "UP" if z < 0 else "DOWN"
                if obs_dir == pred_dir:
                    n_dir_match += 1
                pct = float(sp_stats.percentileofscore(kd_abs_z[up], abs(z))) if up in kd_abs_z else 50.0
                percentiles.append(pct)
                if obs_dir == pred_dir and pct >= PERCENTILE_PARTIAL:
                    n_supported += 1
            else:
                percentiles.append(25.0)
        return {
            "n_supported": n_supported,
            "mean_percentile": np.mean(percentiles) if percentiles else 0,
            "n_direction_match": n_dir_match,
            "n_testable": n_testable,
        }

    # Step 5: Score user's claims
    steps.append("Step 4: Scoring your claims...")
    user_ups = claims["upstream_gene"].tolist()
    user_downs = claims["downstream_gene"].tolist()
    user_dirs = claims["predicted_direction"].tolist()
    user_score = score_set(user_ups, user_downs, user_dirs)
    steps.append(
        f"  Your claims: {user_score['n_supported']}/{user_score['n_testable']} supported, "
        f"mean percentile {user_score['mean_percentile']:.1f}"
    )

    # Step 6: Run permutations
    steps.append(f"Step 5: Running {n_permutations} permutations...")
    np.random.seed(RANDOM_SEED + 2)
    n_unique_ups = len(set(user_ups))
    null_supported = []
    null_percentiles = []
    null_dir_matches = []

    for _i in range(n_permutations):
        random_pool_sample = np.random.choice(pool_genes, n_unique_ups, replace=False)
        upstream_map = dict(zip(sorted(set(user_ups)), random_pool_sample, strict=False))
        random_ups = [upstream_map.get(u, np.random.choice(pool_genes)) for u in user_ups]
        perm_score = score_set(random_ups, user_downs, user_dirs)
        null_supported.append(perm_score["n_supported"])
        null_percentiles.append(perm_score["mean_percentile"])
        null_dir_matches.append(perm_score["n_direction_match"])

    null_supported = np.array(null_supported)
    null_percentiles = np.array(null_percentiles)
    null_dir_matches = np.array(null_dir_matches)

    p_supported = float(np.mean(null_supported >= user_score["n_supported"]))
    p_percentile = float(np.mean(null_percentiles >= user_score["mean_percentile"]))
    p_direction = float(np.mean(null_dir_matches >= user_score["n_direction_match"]))

    # Step 7: Report
    steps.append("Step 6: Specificity test complete.")
    steps.append("")
    steps.append("RESULTS:")
    steps.append(f"  Comparison pool: {pool_source}")
    steps.append(f"  Permutations: {n_permutations}")
    steps.append("")

    steps.append(
        f"  Supported claims: {user_score['n_supported']}/{user_score['n_testable']} "
        f"(null: {np.mean(null_supported):.1f} +/- {np.std(null_supported):.1f}, "
        f"p={p_supported:.4f})"
    )
    if p_supported < 0.05:
        steps.append("    --> Your regulators produce MORE supported claims than random genes.")
    elif p_supported > 0.5:
        steps.append("    --> Random genes score as well or better. Your set is NOT special.")
    else:
        steps.append("    --> No significant difference from random.")

    steps.append(
        f"  Mean effect percentile: {user_score['mean_percentile']:.1f} "
        f"(null: {np.mean(null_percentiles):.1f} +/- {np.std(null_percentiles):.1f}, "
        f"p={p_percentile:.4f})"
    )

    steps.append(
        f"  Direction matches: {user_score['n_direction_match']}/{user_score['n_testable']} "
        f"(null: {np.mean(null_dir_matches):.1f}, p={p_direction:.4f})"
    )

    steps.append("")
    if p_supported < 0.05 and p_percentile < 0.05:
        steps.append(
            "INTERPRETATION: Your upstream regulators are specifically enriched for "
            "these downstream targets -- both in number of supported claims and "
            "effect strength. The regulatory relationships are not explained by "
            "generic genome properties."
        )
    elif p_supported < 0.05:
        steps.append(
            "INTERPRETATION: More of your claims are supported than expected by "
            "chance, but the effect strengths are not exceptional. The wiring is "
            "real but the effects are modest."
        )
    elif p_supported > 0.2:
        steps.append(
            "INTERPRETATION: Random genes score as well as your chosen regulators. "
            "The regulatory relationships may reflect generic properties of the "
            "knockdown dataset rather than specific biology."
        )
    else:
        steps.append(
            "INTERPRETATION: Results are borderline. More claims are supported than "
            "the null average, but not significantly so. Consider testing with a "
            "stricter comparison pool (e.g., disease-associated genes only)."
        )

    # Save results
    result_dict = {
        "user_n_supported": user_score["n_supported"],
        "user_mean_percentile": round(user_score["mean_percentile"], 1),
        "user_n_direction_match": user_score["n_direction_match"],
        "user_n_testable": user_score["n_testable"],
        "null_supported_mean": round(float(np.mean(null_supported)), 2),
        "null_supported_std": round(float(np.std(null_supported)), 2),
        "null_percentile_mean": round(float(np.mean(null_percentiles)), 1),
        "null_percentile_std": round(float(np.std(null_percentiles)), 1),
        "p_supported": round(p_supported, 4),
        "p_percentile": round(p_percentile, 4),
        "p_direction": round(p_direction, 4),
        "n_permutations": n_permutations,
        "pool_source": pool_source,
        "pool_size": len(pool_genes),
    }

    import json

    out_path = os.path.join(output_folder, "truthseq_specificity_results.json")
    with open(out_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    steps.append(f"Detailed results saved to {out_path}")

    return "\n".join(steps)
