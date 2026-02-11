#!/usr/bin/env python3
"""
End-to-end agent integration tests (mocked LLM) for all 24 questions in Tests.pdf.
Validates that the A1 agent executes tool calls and produces expected results.
"""
from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from biomni.agent import A1
import biomni.agent.a1 as a1_module
import biomni.llm as llm_module

CSV_PATH = "/home/oem/Desktop/amplicon-repo-agentai/biomni/data/biomni_data/data_lake/CCLE.csv"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


CASES = [
    {
        "id": "Q1",
        "prompt": "Is YES1 amplified as ecDNA or BFB in CCLE? In which samples?",
        "expects": ["Q1_OK=True"],
    },
    {
        "id": "Q2",
        "prompt": "Summarize the size distribution of all BFB amplifications (Captured interval length)",
        "expects": ["Q2_OK=True"],
    },
    {
        "id": "Q3",
        "prompt": "List all BFB amplifications in HARA (feature IDs, loci, genes, copy number)",
        "expects": ["Q3_OK=True"],
    },
    {
        "id": "Q4",
        "prompt": "Which genes are most frequently amplified as BFB? (top 25)",
        "expects": ["Q4_OK=True"],
    },
    {
        "id": "Q5",
        "prompt": "Show the distribution of amplification classes across cancer types (Tissue of origin)",
        "expects": ["Q5_OK=True"],
    },
    {
        "id": "Q6",
        "prompt": "For each tissue, what fraction of samples have any ecDNA?",
        "expects": ["Q6_OK=True"],
    },
    {
        "id": "Q7",
        "prompt": "Do Complexity scores differ between ecDNA vs BFB vs Linear vs Complex-non-cyclic?",
        "expects": ["Q7_OK=True"],
    },
    {
        "id": "Q8",
        "prompt": "For ecDNA vs BFB, how does size relate to max copy number? (scatter + correlation)",
        "expects": ["Q8_OK=True"],
    },
    {
        "id": "Q9",
        "prompt": "What are the highest copy-number amplifications in CCLE? (top 5)",
        "expects": ["Q9_OK=True"],
    },
    {
        "id": "Q10",
        "prompt": "Which oncogenes appear in both ecDNA and BFB across CCLE?",
        "expects": ["Q10_OK=True"],
    },
    {
        "id": "Q12",
        "prompt": "Which samples contain both ecDNA and BFB amplifications?",
        "expects": ["Q12_OK=True"],
    },
    {
        "id": "Q13",
        "prompt": "Are ecDNA amplifications larger than BFB on average?",
        "expects": ["Q13_OK=True"],
    },
    {
        "id": "Q14",
        "prompt": "Which samples have high-copy (>20 CN) ecDNA?",
        "expects": ["Q14_OK=True"],
    },
    {
        "id": "Q15",
        "prompt": "What is the largest amplicon (by size) per class?",
        "expects": ["Q15_OK=True"],
    },
    {
        "id": "Q16",
        "prompt": "Which tissues show highest amplification complexity (median)?",
        "expects": ["Q16_OK=True"],
    },
    {
        "id": "Q17",
        "prompt": "For a given gene, list all co-amplified genes (same feature)",
        "expects": ["Q17_OK=True"],
    },
    {
        "id": "Q18",
        "prompt": "Which genes are exclusive to ecDNA (never BFB or Linear)?",
        "expects": ["Q18_OK=True"],
    },
    {
        "id": "Q19",
        "prompt": "Which chromosomes are most frequently amplified?",
        "expects": ["Q19_OK=True"],
    },
    {
        "id": "Q20",
        "prompt": "Which oncogenes are most recurrent across unique samples (any class)?",
        "expects": ["Q20_OK=True"],
    },
    {
        "id": "Q21",
        "prompt": "For each tissue, what are the top 5 amplified oncogenes (any class, by #unique samples)?",
        "expects": ["Q21_OK=True"],
    },
    {
        "id": "Q22",
        "prompt": "Which tissues have the highest fraction of BFB features?",
        "expects": ["Q22_OK=True"],
    },
    {
        "id": "Q23",
        "prompt": "Among samples with ecDNA, what are the most common co-amplified oncogene pairs?",
        "expects": ["Q23_OK=True"],
    },
    {
        "id": "Q24",
        "prompt": "Which samples have amplifications on multiple chromosomes? (by feature loci)",
        "expects": ["Q24_OK=True"],
    },
    {
        "id": "Q25",
        "prompt": "For each class, what is the distribution of number of oncogenes per feature?",
        "expects": ["Q25_OK=True"],
    },
]


def _base_code() -> str:
    return f"""
import pandas as pd
import numpy as np
import ast
from biomni.tool.amplicon_table import query_amplicons

CSV_PATH = r"{CSV_PATH}"

def _parse_list_cell(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        raw = x
    else:
        try:
            raw = ast.literal_eval(str(x))
        except Exception:
            return []
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        s = str(item).strip()
        s = s.strip("'").strip('"').strip()
        if s and s.lower() != "nan":
            out.append(s)
    return out

def _df(select_cols=None, **kwargs):
    res = query_amplicons(csv_path=CSV_PATH, select=select_cols, **kwargs)
    return pd.DataFrame(res["rows"])
""".strip()


def _execute_code(case_id: str) -> str:
    base = _base_code()

    if case_id == "Q1":
        body = """
df = _df(select_cols=["Sample name","Classification"], gene="YES1", gene_field="either", classification=["BFB","ecDNA"])
samples = sorted(set(df["Sample name"].tolist()))
ok = ("KYSE70_OESOPHAGUS" in samples) and (len(df) == 1)
print("Q1_OK=", ok)
print("Q1_SAMPLES=", samples)
"""
    elif case_id == "Q2":
        body = """
df = _df(select_cols=["Classification","Captured interval length"], classification="BFB")
sizes = pd.to_numeric(df["Captured interval length"], errors="coerce")
mean = sizes.mean()
median = sizes.median()
ok = (len(df) == 114) and (2.2e6 < mean < 2.3e6) and (1.2e6 < median < 1.3e6)
print("Q2_OK=", ok)
print("Q2_MEAN=", mean)
"""
    elif case_id == "Q3":
        body = """
df = _df(select_cols=["Sample name","Classification","AA amplicon number","Feature ID","Location","Captured interval length","Feature median copy number","Feature maximum copy number","Oncogenes","All genes"], classification="BFB")
bt = df[df["Sample name"] == "HARA_LUNG"]
ok = len(bt) == 1
print("Q3_OK=", ok)
print("Q3_COUNT=", len(bt))
"""
    elif case_id == "Q4":
        body = """
df = _df(select_cols=["Classification","Oncogenes"], classification="BFB")
df["oncogenes_list"] = df["Oncogenes"].apply(_parse_list_cell)
from collections import Counter
ctr = Counter()
for genes in df["oncogenes_list"]:
    ctr.update(set(genes))
top_gene, top_count = ctr.most_common(1)[0]
ok = (top_gene == "CTTN" and top_count == 5)
print("Q4_OK=", ok)
print("Q4_TOP1=", top_gene, top_count)
"""
    elif case_id == "Q5":
        body = """
df = _df(select_cols=["Tissue of origin","Classification"])
tab = pd.crosstab(df["Tissue of origin"], df["Classification"])
breast = tab.loc["breast"]
ok = (breast["BFB"] == 13 and breast["Complex-non-cyclic"] == 32 and breast["Linear"] == 101 and breast["ecDNA"] == 62)
print("Q5_OK=", ok)
print("Q5_BREAST=", dict(breast))
"""
    elif case_id == "Q6":
        body = """
df = _df(select_cols=["Tissue of origin","Sample name","Classification"])
sample_has_ec = (
    df.assign(has_ecDNA=df["Classification"].eq("ecDNA"))
    .groupby(["Tissue of origin","Sample name"])["has_ecDNA"]
    .any()
    .reset_index()
)
tissue_frac = (
    sample_has_ec.groupby("Tissue of origin")["has_ecDNA"]
    .mean()
    .sort_values(ascending=False)
)
top = tissue_frac.head(3).index.tolist()
ok = (tissue_frac.get("thyroid", 0) == 1.0) and (tissue_frac.get("oesophagus", 0) == 1.0)
print("Q6_OK=", ok)
print("Q6_TOP=", top)
"""
    elif case_id == "Q7":
        body = """
df = _df(select_cols=["Classification","Complexity score"]).dropna()
med = df.groupby("Classification")["Complexity score"].median()
ok = (
    abs(med["BFB"] - 1.021014) < 0.02 and
    abs(med["Complex-non-cyclic"] - 1.189101) < 0.02 and
    abs(med["Linear"] - 0.636480) < 0.02 and
    abs(med["ecDNA"] - 0.686838) < 0.02
)
print("Q7_OK=", ok)
print("Q7_MED=", med.to_dict())
"""
    elif case_id == "Q8":
        body = """
df = _df(select_cols=["Classification","Captured interval length","Feature maximum copy number"])
sub = df[df["Classification"].isin(["ecDNA","BFB"])].copy()
sub = sub.dropna(subset=["Captured interval length","Feature maximum copy number"])
def _corr(cls):
    s = sub[sub["Classification"] == cls]
    return s["Captured interval length"].corr(s["Feature maximum copy number"])
ec_corr = _corr("ecDNA")
bfb_corr = _corr("BFB")
ok = abs(ec_corr - 0.0498) < 0.001 and abs(bfb_corr - (-0.0029)) < 0.001
print("Q8_OK=", ok)
print("Q8_CORR=", ec_corr, bfb_corr)
"""
    elif case_id == "Q9":
        body = """
df = _df(select_cols=["Sample name","Tissue of origin","Classification","Feature ID","Location","Feature maximum copy number","Feature median copy number","Captured interval length","Oncogenes"])
top = df.dropna(subset=["Feature maximum copy number"]).sort_values("Feature maximum copy number", ascending=False).head(1)
row = top.iloc[0]
ok = row["Sample name"] == "NCIH524_LUNG" and row["Classification"] == "ecDNA"
print("Q9_OK=", ok)
print("Q9_TOP=", row["Sample name"], row["Feature maximum copy number"])
"""
    elif case_id == "Q10":
        body = """
df = _df(select_cols=["Classification","Oncogenes"])
df["oncogenes_list"] = df["Oncogenes"].apply(_parse_list_cell)
from collections import defaultdict
gene_classes = defaultdict(set)
for _, r in df.iterrows():
    for g in r["oncogenes_list"]:
        gene_classes[g].add(r["Classification"])
both = sorted([g for g, cls in gene_classes.items() if {"ecDNA","BFB"} <= cls])
ok = (len(both) == 67) and ("MYC" in both)
print("Q10_OK=", ok)
print("Q10_COUNT=", len(both))
"""
    elif case_id == "Q12":
        body = """
df = _df(select_cols=["Sample name","Classification"])
sample_classes = df.groupby("Sample name")["Classification"].apply(set)
both = sample_classes[sample_classes.apply(lambda x: {"ecDNA","BFB"} <= x)]
ok = (len(both) == 51) and ("HARA_LUNG" in both.index) and ("KYSE70_OESOPHAGUS" in both.index)
print("Q12_OK=", ok)
print("Q12_COUNT=", len(both))
"""
    elif case_id == "Q13":
        body = """
df = _df(select_cols=["Classification","Captured interval length"])
means = df.groupby("Classification")["Captured interval length"].mean()
ok = means["BFB"] > means["ecDNA"] and (2.2e6 < means["BFB"] < 2.3e6)
print("Q13_OK=", ok)
print("Q13_MEANS=", means.to_dict())
"""
    elif case_id == "Q14":
        body = """
df = _df(select_cols=["Sample name","Location","Feature maximum copy number","Oncogenes","Classification"])
hc = df[(df["Classification"] == "ecDNA") & (df["Feature maximum copy number"] > 20)]
samples = hc["Sample name"].tolist()
ok = (len(hc) > 0) and ("5637_URINARY_TRACT" in samples)
print("Q14_OK=", ok)
print("Q14_COUNT=", len(hc))
"""
    elif case_id == "Q15":
        body = """
df = _df(select_cols=["Classification","Sample name","Location","Captured interval length","Oncogenes"])
largest = df.sort_values("Captured interval length", ascending=False).groupby("Classification").head(1)
lookup = dict(zip(largest["Classification"], largest["Sample name"]))
ok = (
    lookup.get("Complex-non-cyclic") == "SNU119_OVARY" and
    lookup.get("ecDNA") == "HCC1419_BREAST" and
    lookup.get("BFB") == "HEPG2_LIVER" and
    lookup.get("Linear") == "SCC25_UPPER_AERODIGESTIVE_TRACT"
)
print("Q15_OK=", ok)
print("Q15_TOP=", lookup)
"""
    elif case_id == "Q16":
        body = """
df = _df(select_cols=["Tissue of origin","Complexity score"])
tissue_complexity = df.groupby("Tissue of origin")["Complexity score"].median().sort_values(ascending=False)
top3 = tissue_complexity.head(3).index.tolist()
ok = all(t in top3 for t in ["pleura","autonomic ganglia","soft tissue"])
print("Q16_OK=", ok)
print("Q16_TOP3=", top3)
"""
    elif case_id == "Q17":
        body = """
df = _df(select_cols=["All genes"])
df["all_genes_list"] = df["All genes"].apply(_parse_list_cell)
gene = "EGFR"
co = set()
for xs in df["all_genes_list"]:
    if gene in xs:
        co |= set(xs)
co.discard(gene)
ok = "SEC61G" in co
print("Q17_OK=", ok)
print("Q17_COUNT=", len(co))
"""
    elif case_id == "Q18":
        body = """
df = _df(select_cols=["Classification","Oncogenes"])
df["oncogenes_list"] = df["Oncogenes"].apply(_parse_list_cell)
from collections import defaultdict
gene_cls = defaultdict(set)
for _, r in df.iterrows():
    for g in r["oncogenes_list"]:
        gene_cls[g].add(r["Classification"])
exclusive = sorted([g for g, c in gene_cls.items() if c == {"ecDNA"}])
ok = ("MYCN" in exclusive) and ("HMGA2" in exclusive)
print("Q18_OK=", ok)
print("Q18_COUNT=", len(exclusive))
"""
    elif case_id == "Q19":
        body = """
df = _df(select_cols=["Location"])
chroms = df["Location"].str.extract(r"(chr\d+|chrX|chrY)", expand=False)
chrom_counts = chroms.value_counts()
top_chr = chrom_counts.index[0]
top_count = int(chrom_counts.iloc[0])
ok = (top_chr == "chr1" and top_count == 122)
print("Q19_OK=", ok)
print("Q19_TOP=", top_chr, top_count)
"""
    elif case_id == "Q20":
        body = """
df = _df(select_cols=["Sample name","Oncogenes"])
df["oncogenes_list"] = df["Oncogenes"].apply(_parse_list_cell)
pairs = []
for _, r in df.iterrows():
    for g in set(r["oncogenes_list"]):
        pairs.append((g, r["Sample name"]))
tmp = pd.DataFrame(pairs, columns=["Gene","Sample"]).drop_duplicates()
top = tmp.groupby("Gene")["Sample"].nunique().sort_values(ascending=False)
ok = (top["PVT1"] == 33) and (top["MYC"] == 33)
print("Q20_OK=", ok)
print("Q20_TOP=", top.head(2).to_dict())
"""
    elif case_id == "Q21":
        body = """
df = _df(select_cols=["Tissue of origin","Sample name","Oncogenes"])
df["oncogenes_list"] = df["Oncogenes"].apply(_parse_list_cell)
rows = []
for _, r in df.iterrows():
    t = r["Tissue of origin"]
    s = r["Sample name"]
    for g in set(r["oncogenes_list"]):
        rows.append((t, g, s))
x = pd.DataFrame(rows, columns=["Tissue","Gene","Sample"]).drop_duplicates()
breast = x[x["Tissue"] == "breast"]
top = breast.groupby("Gene")["Sample"].nunique().sort_values(ascending=False)
ok = (top["ERBB2"] == 10) and (top["MIEN1"] == 10)
print("Q21_OK=", ok)
print("Q21_TOP=", top.head(5).to_dict())
"""
    elif case_id == "Q22":
        body = """
df = _df(select_cols=["Tissue of origin","Classification"])
tab = pd.crosstab(df["Tissue of origin"], df["Classification"])
frac_bfb = (tab["BFB"] / tab.sum(axis=1)).sort_values(ascending=False)
top = frac_bfb.head(3)
ok = (abs(top.iloc[0] - 0.5) < 1e-6) and (top.index[0] == "bone")
print("Q22_OK=", ok)
print("Q22_TOP3=", top.to_dict())
"""
    elif case_id == "Q23":
        body = """
df = _df(select_cols=["Classification","Oncogenes"])
df["oncogenes_list"] = df["Oncogenes"].apply(_parse_list_cell)
ec = df[df["Classification"] == "ecDNA"]
from itertools import combinations
from collections import Counter
pair_ctr = Counter()
for xs in ec["oncogenes_list"]:
    genes = sorted(set(xs))
    for a, b in combinations(genes, 2):
        pair_ctr[(a, b)] += 1
top_pair, top_count = pair_ctr.most_common(1)[0]
ok = (top_pair == ("MYC", "PVT1") and top_count == 23)
print("Q23_OK=", ok)
print("Q23_TOP=", top_pair, top_count)
"""
    elif case_id == "Q24":
        body = """
df = _df(select_cols=["Sample name","Location"])
chrom = df["Location"].str.extract(r"(chr\d+|chrX|chrY)", expand=False)
tmp = df.copy()
tmp["chrom"] = chrom
per_sample_chr = (
    tmp.dropna(subset=["chrom"])
    .groupby("Sample name")["chrom"]
    .nunique()
    .sort_values(ascending=False)
)
top_samples = per_sample_chr.head(5)
ok = top_samples.max() >= 7
print("Q24_OK=", ok)
print("Q24_TOP=", top_samples.to_dict())
"""
    elif case_id == "Q25":
        body = """
df = _df(select_cols=["Classification","Oncogenes"])
df["oncogenes_list"] = df["Oncogenes"].apply(_parse_list_cell)
df["oncogene_count"] = df["oncogenes_list"].apply(lambda xs: len(set(xs)))
stats = df.groupby("Classification")["oncogene_count"].mean()
ok = (
    abs(stats["BFB"] - 1.412281) < 0.05 and
    abs(stats["Complex-non-cyclic"] - 1.897959) < 0.05 and
    abs(stats["Linear"] - 0.532727) < 0.05 and
    abs(stats["ecDNA"] - 1.727273) < 0.05
)
print("Q25_OK=", ok)
print("Q25_MEAN=", stats.to_dict())
"""
    else:
        body = """print('UNKNOWN_CASE')"""

    return f"{base}\n\n{body.strip()}"


class MockLLM:
    """Mock LLM that emits an <execute> block for each prompt, then <solution>."""

    model_name = "mock-llm"

    def __init__(self) -> None:
        self._answered = False

    def invoke(self, messages: list[Any]) -> AIMessage:
        if any(isinstance(m, AIMessage) and "<observation>" in m.content for m in messages):
            return AIMessage(content="<solution>Completed mocked agent run.</solution>")

        prompt = None
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                prompt = m.content
                break

        case = next((c for c in CASES if c["prompt"] == prompt), None)
        if case is None:
            return AIMessage(content="<solution>Unknown prompt.</solution>")

        code = _execute_code(case["id"])
        return AIMessage(content=f"<execute>\n{code}\n</execute>")

    def with_structured_output(self, output_class):  # noqa: D401
        return self


def run_agent_integration_tests() -> None:
    _assert(os.path.exists(CSV_PATH), f"CCLE.csv not found at {CSV_PATH}")

    original_get_llm = llm_module.get_llm
    original_get_llm_a1 = a1_module.get_llm
    llm_module.get_llm = lambda *args, **kwargs: MockLLM()
    a1_module.get_llm = lambda *args, **kwargs: MockLLM()

    try:
        for case in CASES:
            print(f"\n=== Running {case['id']} ===")
            agent = A1(llm="mock", expected_data_lake_files=[], use_tool_retriever=False)
            log, final_content = agent.go(case["prompt"])
            combined_log = "\n".join(log)
            normalized_log = combined_log.replace("OK= True", "OK=True")

            for expected in case["expects"]:
                _assert(expected in normalized_log, f"{case['id']} missing expected output: {expected}")

            _assert("<solution>" in final_content, f"{case['id']} did not produce solution")

        print("\n✓ All 24 agent integration cases passed")
    finally:
        llm_module.get_llm = original_get_llm
        a1_module.get_llm = original_get_llm_a1


if __name__ == "__main__":
    run_agent_integration_tests()
