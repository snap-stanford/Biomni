from biomni.eval.biomni_eval1 import BiomniEval1


def test_patient_gene_detection_normalizes_gene_symbol_case():
    evaluator = object.__new__(BiomniEval1)

    assert evaluator._compute_reward("patient_gene_detection", '{"causal_gene": [" brca1 "]}', "BRCA1, TP53") == 1.0


def test_patient_gene_detection_keeps_nonmatching_gene_incorrect():
    evaluator = object.__new__(BiomniEval1)

    assert evaluator._compute_reward("patient_gene_detection", '{"causal_gene": ["EGFR"]}', "BRCA1") == 0.0
