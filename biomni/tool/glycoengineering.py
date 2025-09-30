"""
Glycoengineering tools: quick, dependency-light utilities for glycosylation analysis
and curated links to external, specialized software referenced in issue #198.

Functions return research-log style strings to match Biomni tool patterns.
"""


def find_n_glycosylation_motifs(sequence: str, allow_overlap: bool = False) -> str:
    """Scan a protein sequence for N-linked glycosylation sequons (N-X-[S/T]).

    Rules
    - Motif: Asn (N) followed by any residue except Proline (P), followed by Serine (S) or Threonine (T)
    - By default, overlapping matches are not reported twice

    Parameters
    - sequence: protein sequence (one-letter amino-acid codes)
    - allow_overlap: if True, allow overlapping motif detection

    Returns
    - Research log string summarizing motif locations and counts
    """
    seq = (sequence or "").upper()
    results: list[dict] = []

    i = 0
    while i <= len(seq) - 3:
        tri = seq[i : i + 3]
        if tri[0] == "N" and tri[1] != "P" and tri[2] in {"S", "T"}:
            results.append({"position": i + 1, "motif": tri})  # 1-based
            i = i + (1 if allow_overlap else 3)
        else:
            i += 1

    log = ["# N-linked glycosylation sequon scan (N-X-[S/T], X≠P)"]
    log.append(f"Sequence length: {len(seq)}")
    log.append(f"Total sequons found: {len(results)}")
    if results:
        log.append("\nPositions (1-based):")
        for r in results[:100]:  # print at most 100 entries to keep output manageable
            log.append(f"- {r['position']}: {r['motif']}")
        if len(results) > 100:
            log.append(f"... and {len(results) - 100} more")
    else:
        log.append("No canonical N-linked sequons detected.")
    return "\n".join(log)


def predict_o_glycosylation_hotspots(
    sequence: str,
    window: int = 7,
    min_st_fraction: float = 0.4,
    disallow_proline_next: bool = True,
) -> str:
    """Heuristic O-glycosylation hotspot scoring.

    Background
    - O-GalNAc glycosylation frequently occurs on Ser/Thr-rich segments.
    - This lightweight heuristic flags residues in local windows enriched for S/T.
    - Not a substitute for NetOGlyc; provided as a fast, dependency-free baseline.

    Parameters
    - sequence: protein sequence (one-letter AA codes)
    - window: odd window size for local S/T density (default 7)
    - min_st_fraction: minimum S/T fraction in window to flag sites (0..1)
    - disallow_proline_next: if True, avoid flagging S/T immediately followed by Proline

    Returns
    - Research log string with candidate sites and scores
    """
    if window < 3 or window % 2 == 0:
        window = 7
    seq = (sequence or "").upper()
    half = window // 2

    candidates: list[dict] = []
    for i, aa in enumerate(seq):
        if aa not in {"S", "T"}:
            continue
        start = max(0, i - half)
        end = min(len(seq), i + half + 1)
        segment = seq[start:end]
        st_count = sum(1 for c in segment if c in {"S", "T"})
        frac = st_count / max(1, len(segment))
        if disallow_proline_next and i + 1 < len(seq) and seq[i + 1] == "P":
            continue
        if frac >= min_st_fraction:
            candidates.append(
                {
                    "position": i + 1,
                    "residue": aa,
                    "st_fraction": round(frac, 3),
                    "window": f"{start + 1}-{end}",
                }
            )

    log = ["# Heuristic O-glycosylation hotspot prediction (S/T density based)"]
    log.append(f"Sequence length: {len(seq)} | window={window} | threshold={min_st_fraction}")
    log.append(f"Total candidate sites: {len(candidates)}")
    if candidates:
        log.append("\nTop candidates:")
        for c in candidates[:100]:
            log.append(
                f"- pos {c['position']} ({c['residue']}): S/T fraction={c['st_fraction']} in window {c['window']}"
            )
        if len(candidates) > 100:
            log.append(f"... and {len(candidates) - 100} more")
    else:
        log.append("No candidate O-glycosylation hotspots met the heuristic threshold.")

    log.append("\nNote: For state-of-the-art O-glycosite prediction, use NetOGlyc 4.0 (web service).")
    return "\n".join(log)


def list_glycoengineering_resources() -> str:
    """Curate and summarize external glycoengineering tools and resources.

    Includes links referenced in issue #198 and brief usage notes.
    Returns a research-log style summary with URLs for further use.
    """
    lines = ["# Glycoengineering tools and resources (curated)"]
    lines.append("")
    lines.append("1) Glycoshield-md")
    lines.append("   - MD workflows for glycan shielding analysis around proteins.")
    lines.append("   - URL: https://gitlab.mpcdf.mpg.de/dioscuri-biophysics/glycoshield-md/")
    lines.append("")
    lines.append("2) SweetTalk")
    lines.append("   - Language-model based framework applied to glycomics/glycoproteomics tasks.")
    lines.append("   - Search on GitHub for installation and examples (various forks exist).")
    lines.append("")
    lines.append("3) N-Glycosylation Markov Models")
    lines.append("   - Probabilistic modeling of N-glycan biosynthesis/patterns.")
    lines.append("   - Search GitHub for ‘N-Glycosylation-Markov-Models’ implementations.")
    lines.append("")
    lines.append("4) Copenhagen Center for Glycomics (CCG)")
    lines.append("   - Consortium hosting many glyco-related datasets, tools, and protocols.")
    lines.append("   - URL: https://github.com/CopenhagenCenterForGlycomics")
    lines.append("")
    lines.append("5) NetOGlyc 4.0 (O-glycosite predictor)")
    lines.append("   - Web service for O-GalNAc site prediction; check license/usage terms.")
    lines.append("   - URL: https://services.healthtech.dtu.dk/services/NetOGlyc-4.0/")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Many tools are heavy and best run via their own environments or containers.")
    lines.append("- For tight Biomni integration, consider adding MCP wrappers or CLI installers.")
    return "\n".join(lines)
