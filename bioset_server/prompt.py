# prompt.py

PROMPT_TEMPLATE = """
You are a domain expert in tumor immunology and CyCIF (cyclic immunofluorescence) analysis, with experience interpreting marker co-expression (cell identity/state) and spatial neighborhoods (cell–cell interaction) in {DATASET}.

I am analyzing a 3D {DATASET} dataset and want biologically meaningful labels for:
(A) marker CO-LOCALIZATION / CO-EXPRESSION patterns that imply a specific cell type or cellular state, and
(B) marker PROXIMITY / NEIGHBORHOOD interactions that imply functional interaction between different cell populations (e.g., immune–tumor interface).

INPUT SPECIFICATION
You will receive a single JSON object in the following format:

{{
  "label": ["MarkerA", "MarkerB", "MarkerC", ...]
}}

- "label" contains a list of biomarker channel names.
- Marker names are case-sensitive and must be used exactly as provided.
- The list may contain one or more markers.

WHAT "+" AND "/" MEAN (CRITICAL)
- "+" means CO-LOCALIZATION / CO-EXPRESSION in the same cells (phenotype/state definition).
- "/" means INTERACTION / NEIGHBORHOOD adjacency between different cell populations.

IMPORTANT: Do NOT choose only one interpretation when both are meaningful.
If a co-localized phenotype is meaningful AND its interaction with another marker/population is meaningful, output BOTH:
- the co-localization key (with "+"), AND
- the interaction key (with "/") that uses that phenotype as one side of the interaction.

KEY GRAMMAR (HOW TO WRITE KEYS)
1) Single marker:
   - Key is the marker name itself (e.g., "CD8A").

2) Co-localization phenotype (same cells):
   - Use "+" to join markers that co-express to define a phenotype/state (e.g., "CD3+CD8A").
   - Markers within a "+" group must be alphabetically ordered.

3) Interaction / neighborhood (different cells nearby):
   - Use "/" to join TWO OR MORE groups that interact spatially.
   - Each side of "/" can be either:
     - a single marker, OR
     - a "+" group representing a phenotype.
   - Examples:
     - "CD3+FOXP3/MART1"
     - "CD8A/PDL1"
     - "CD3+CD8A/MART1+PDL1" (only if biologically meaningful)
   - Within each "+" group: alphabetical order.
   - For "/" interactions, sort the groups alphabetically by their string representation.

SCOPE: WHICH KEYS TO OUTPUT
Given the input marker list, output labels for:
A) All meaningful CO-LOCALIZATION phenotype keys.
B) All meaningful INTERACTION keys, including interactions where one side is a co-localized phenotype group.
You do NOT need to output every mathematically possible combination if it has no biological meaning. Prefer high-signal outputs.

LABEL VALUE FORMAT (LIST)
Every output value MUST be a list of strings:
- The first element is the primary label (title).
- Additional elements (subtitle(s)) are OPTIONAL and should be rare.

CRITICAL RULE FOR SUBTITLES (STRICT)
Only include additional label elements (subtitles) when they add genuinely NEW biological meaning that is not already implied by the title and is useful for interpretation or disambiguation in a visualization.

- Good (adds disambiguation / contrast):
  "PRAME": ["melanoma antigen", "not melanocyte"]

- Bad (redundant / restates the title):
  "CD103": ["tissue-resident memory T cell", "TRM marker"]

Do NOT add subtitles that are merely synonyms, expansions, or “X marker” restatements of the title.
When in doubt, omit the subtitle and output a single-element list.

LABEL TEXT RULES (STRICT)
- Labels must be concise and suitable for scientific visualization legends.
- Do NOT include the words "marker" or "markers" anywhere in any label element.
- Prefer short noun phrases, not sentences.

INTERACTION LABEL GUIDANCE
For interaction labels, use biologically meaningful terminology such as:
interface, niche, zone, border, front, contact, synapse, neighborhood, adjacency, margin, microenvironment (not limited to these words).

WHEN TO RETURN "None"
If no meaningful biological interpretation exists for a key, return:
["None"]

SPECIAL KEY
Include a special key:
"overall"
This represents the general microenvironment or neighborhood context implied by the entire provided marker set.
Its value must also be a list.

OUTPUT REQUIREMENTS (STRICT)
Return ONLY a single valid JSON object.
Do NOT include explanations, comments, markdown, or any other text outside the JSON.

OUTPUT FORMAT EXAMPLE
{{
  "PRAME": ["melanoma antigen", "not melanocyte"],
  "CD3": ["pan–T cell"],
  "FOXP3": ["regulatory T cell"],
  "CD3+FOXP3": ["regulatory T cell"],
  "CD3+FOXP3/MART1": ["Treg–tumor interface"],
  "overall": ["immune–tumor niche"]
}}

If uncertain, return ["None"] rather than speculating.
"""

def make_prompt(dataset: str) -> str:
    dataset = (dataset or "").strip()
    if not dataset:
        dataset = "melanoma CyCIF"
    return PROMPT_TEMPLATE.format(DATASET=dataset)