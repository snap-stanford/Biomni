"""Offline checks that FDA severity and outcome filters keep reports matching any listed value."""

import copy
import json
import unittest
from unittest.mock import patch

import requests
from biomni.tool.pharmacology import _apply_fda_filters, query_fda_adverse_events

RECORDS = [
    {"safetyreportid": "death", "serious": "1", "seriousnessdeath": "1"},
    {"safetyreportid": "hospitalization", "serious": "1", "seriousnesshospitalization": "1"},
    {"safetyreportid": "life_threatening", "serious": "1", "seriousnesslifethreatening": "1"},
    {"safetyreportid": "non_serious", "serious": "2"},
]


def kept(filters):
    response = {"meta": {"results": {"total": len(RECORDS)}}, "results": copy.deepcopy(RECORDS)}
    filtered = _apply_fda_filters(response, filters)
    ids = [row["safetyreportid"] for row in filtered["results"]]
    assert filtered["meta"]["results"]["total"] == len(ids)
    return ids


class FDAOutcomeFilterTest(unittest.TestCase):
    def test_two_outcomes_keep_either(self):
        self.assertEqual(kept({"outcome_filter": ["death", "hospitalization"]}), ["death", "hospitalization"])

    def test_all_documented_outcomes(self):
        self.assertEqual(
            kept({"outcome_filter": ["life_threatening", "hospitalization", "death"]}),
            ["death", "hospitalization", "life_threatening"],
        )

    def test_single_outcome(self):
        self.assertEqual(kept({"outcome_filter": ["death"]}), ["death"])

    def test_both_severity_levels_keep_all(self):
        self.assertEqual(kept({"severity_filter": ["serious", "non_serious"]}), [r["safetyreportid"] for r in RECORDS])

    def test_single_severity_level(self):
        self.assertEqual(kept({"severity_filter": ["serious"]}), ["death", "hospitalization", "life_threatening"])
        self.assertEqual(kept({"severity_filter": ["non_serious"]}), ["non_serious"])

    def test_public_tool_counts_both_outcomes(self):
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"meta": {"results": {"total": 4}}, "results": RECORDS}).encode()
        with patch("requests.sessions.Session.request", return_value=response):
            summary = query_fda_adverse_events("aspirin", outcome_filter=["death", "hospitalization"])
        self.assertIn("Total Reports: 2\n", summary)
        self.assertIn("- Deaths: 1 (50.0%)", summary)
        self.assertIn("- Hospitalizations: 1 (50.0%)", summary)


if __name__ == "__main__":
    unittest.main()
