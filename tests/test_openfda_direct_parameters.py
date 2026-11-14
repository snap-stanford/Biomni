"""Offline checks that documented direct OpenFDA arguments reach the real HTTP client."""

import json
import unittest
from unittest.mock import patch

import requests
from biomni.tool.database import query_openfda


class OpenFDADirectParametersTest(unittest.TestCase):
    def setUp(self):
        self.payload = {"meta": {"results": {"total": 1}}, "results": [{"safetyreportid": "1"}]}
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(self.payload).encode()
        request_patch = patch("requests.sessions.Session.request", return_value=response)
        self.request = request_patch.start()
        self.addCleanup(request_patch.stop)

    def query(self, **kwargs):
        result = query_openfda(endpoint="/drug/event.json", **kwargs)
        self.assertTrue(result["success"])
        self.assertEqual(result["result"], self.payload)
        self.assertEqual(self.request.call_count, 1)
        self.assertEqual(self.request.call_args.kwargs["method"].upper(), "GET")
        self.assertEqual(self.request.call_args.kwargs["url"], "https://api.fda.gov/drug/event.json?limit=100")
        return self.request.call_args.kwargs.get("params") or {}

    def test_documented_drug_search(self):
        params = self.query(search_params={"patient.drug.medicinalproduct": "lipitor"})
        self.assertEqual(params.get("search"), 'patient.drug.medicinalproduct:"lipitor"')

    def test_list_terms_and_multiple_fields(self):
        params = self.query(search_params={"patient.drug.medicinalproduct": ["lipitor", "aspirin"], "serious": "1"})
        self.assertEqual(params.get("search"), 'patient.drug.medicinalproduct:("lipitor" OR "aspirin") AND serious:"1"')

    def test_documented_count_query(self):
        params = self.query(count_params="patient.reaction.reactionmeddrapt.exact")
        self.assertEqual(params.get("count"), "patient.reaction.reactionmeddrapt.exact")

    def test_sort_and_pagination(self):
        params = self.query(sort_params={"receivedate": "desc"}, skip_results=1)
        self.assertEqual(params.get("sort"), "receivedate:desc")
        self.assertEqual(params.get("skip"), 1)

    def test_no_options_preserves_default_query(self):
        self.assertEqual(self.query(), {})

    def test_full_url_with_existing_search_is_preserved(self):
        endpoint = "https://api.fda.gov/drug/event.json?search=serious:1&limit=5"
        result = query_openfda(endpoint=endpoint)
        self.assertTrue(result["success"])
        self.assertEqual(self.request.call_args.kwargs["method"].upper(), "GET")
        self.assertEqual(self.request.call_args.kwargs["url"], endpoint)
        self.assertFalse(self.request.call_args.kwargs.get("params"))


if __name__ == "__main__":
    unittest.main()
