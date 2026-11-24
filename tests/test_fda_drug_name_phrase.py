"""Offline checks that OpenFDA drug-name searches match multi-word names as a phrase."""

import json
import unittest
from unittest.mock import patch

import requests
from biomni.tool.pharmacology import (
    OpenFDAClient,
    check_fda_drug_recalls,
    get_fda_drug_label_info,
    query_fda_adverse_events,
)


class FDADrugNamePhraseTest(unittest.TestCase):
    def setUp(self):
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"meta": {"results": {"total": 0}}, "results": []}).encode()
        request_patch = patch("requests.sessions.Session.request", return_value=response)
        self.request = request_patch.start()
        self.addCleanup(request_patch.stop)

    def sent_search(self):
        self.assertEqual(self.request.call_count, 1)
        return self.request.call_args.kwargs["params"]["search"]

    def test_adverse_events_quote_multi_word_name(self):
        query_fda_adverse_events("valproic acid")
        self.assertEqual(self.sent_search(), 'patient.drug.medicinalproduct:"valproic acid"')

    def test_labels_quote_multi_word_name(self):
        get_fda_drug_label_info("insulin glargine")
        self.assertEqual(self.sent_search(), 'openfda.brand_name:"insulin glargine"')

    def test_recalls_quote_multi_word_name(self):
        check_fda_drug_recalls("valproic acid")
        self.assertEqual(self.sent_search(), 'openfda.brand_name:"valproic acid"')

    def test_single_word_name(self):
        OpenFDAClient().query_adverse_events("aspirin", limit=5)
        self.assertEqual(self.sent_search(), 'patient.drug.medicinalproduct:"aspirin"')
        self.assertEqual(self.request.call_args.kwargs["params"]["limit"], 5)


if __name__ == "__main__":
    unittest.main()
