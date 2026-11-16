"""Offline regression checks for the documented recall classification filter."""

import json
import unittest
from unittest.mock import patch

import requests
from biomni.tool.pharmacology import OpenFDAClient, check_fda_drug_recalls


class FDARecallClassificationTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"recall_number": str(index), "classification": classification}
            for index, classification in enumerate(["Class I", "Class II", "Class III"], 1)
        ]
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"meta": {"results": {"total": 3}}, "results": self.records}).encode()
        request_patch = patch("requests.sessions.Session.request", return_value=response)
        self.request = request_patch.start()
        self.addCleanup(request_patch.stop)

    def test_public_tool_excludes_other_classes(self):
        summary = check_fda_drug_recalls("aspirin", classification=["Class I"])
        self.assertIn("Classification filter: Class I", summary)
        self.assertIn("Total recalls found: 1", summary)
        self.assertIn("Classification: Class I\n", summary)
        self.assertNotIn("Classification: Class II", summary)
        self.assertEqual(self.request.call_count, 1)

    def test_client_accepts_multiple_classes(self):
        result = OpenFDAClient().query_drug_recalls("aspirin", classification=["Class I", "Class III"])
        self.assertEqual([row["classification"] for row in result["results"]], ["Class I", "Class III"])
        self.assertEqual(result["meta"]["results"]["total"], 2)

    def test_no_matching_class_returns_no_recalls(self):
        self.records[0]["classification"] = "Class II"
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"meta": {"results": {"total": 3}}, "results": self.records}).encode()
        self.request.return_value = response
        summary = check_fda_drug_recalls("aspirin", classification=["Class I"])
        self.assertEqual(summary, "No drug recalls found for aspirin in the FDA database.")

    def test_no_filter_preserves_response(self):
        for classification in [None, []]:
            with self.subTest(classification=classification):
                result = OpenFDAClient().query_drug_recalls("aspirin", classification=classification)
                self.assertEqual(result["results"], self.records)
                self.assertEqual(result["meta"]["results"]["total"], 3)


if __name__ == "__main__":
    unittest.main()
