"""Offline regression tests for documented OpenFDA report-level outcome flags."""

import json
import unittest
from unittest.mock import patch

import requests
from biomni.tool.pharmacology import _generate_fda_statistics, query_fda_adverse_events


class FDAOutcomeStatisticsTest(unittest.TestCase):
    def test_public_summary_includes_all_three_outcomes(self):
        records = [
            {
                "safetyreportid": str(index),
                "serious": "1",
                field: "1",
                "patient": {"reaction": [{"reactionmeddrapt": "NAUSEA"}]},
            }
            for index, field in enumerate(
                ["seriousnessdeath", "seriousnesslifethreatening", "seriousnesshospitalization"], 1
            )
        ]
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"meta": {"results": {"total": 3}}, "results": records}).encode()
        with patch("requests.sessions.Session.request", return_value=response) as request:
            summary = query_fda_adverse_events("aspirin")
        self.assertEqual(request.call_count, 1)
        self.assertIn("Total Reports: 3", summary)
        for label in ["Deaths", "Life-threatening", "Hospitalizations"]:
            with self.subTest(label=label):
                self.assertIn(f"- {label}: 1 (33.3%)", summary)

    def test_missing_flags_do_not_count(self):
        stats = _generate_fda_statistics({"results": [{}, {"serious": "2", "patient": {"reaction": []}}]})
        for key in ["death_reports", "life_threatening_reports", "hospitalization_reports"]:
            with self.subTest(key=key):
                self.assertEqual(stats[key], 0)

    def test_counts_reports_rather_than_reactions(self):
        stats = _generate_fda_statistics(
            {
                "results": [
                    {
                        "seriousnessdeath": "1",
                        "seriousnesslifethreatening": "1",
                        "seriousnesshospitalization": "1",
                        "patient": {"reaction": [{"reactionmeddrapt": "NAUSEA"}, {"reactionmeddrapt": "HEADACHE"}]},
                    }
                ]
            }
        )
        for key in ["death_reports", "life_threatening_reports", "hospitalization_reports"]:
            with self.subTest(key=key):
                self.assertEqual(stats[key], 1)


if __name__ == "__main__":
    unittest.main()
