"""Offline checks that both query_gnomad paths build and send a GraphQL request."""

import json
import unittest
from unittest.mock import patch

import requests
from biomni.tool import database
from biomni.tool.database import query_gnomad


class GnomadQueryTest(unittest.TestCase):
    def setUp(self):
        self.payload = {"data": {"gene": {"variants": []}}}
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(self.payload).encode()
        request_patch = patch("requests.sessions.Session.request", return_value=response)
        self.request = request_patch.start()
        self.addCleanup(request_patch.stop)

    def sent_query(self):
        self.assertEqual(self.request.call_count, 1)
        self.assertEqual(self.request.call_args.kwargs["method"].upper(), "POST")
        self.assertEqual(self.request.call_args.kwargs["url"], "https://gnomad.broadinstitute.org/api")
        return self.request.call_args.kwargs["json"]["query"]

    def test_direct_gene_symbol(self):
        result = query_gnomad(gene_symbol="TP53")

        self.assertTrue(result["success"])
        self.assertEqual(result["result"], self.payload)
        self.assertEqual(result["query_info"]["description"], "Query gnomAD for variants in TP53")
        query = self.sent_query()
        self.assertIn('gene(gene_symbol: "TP53"', query)
        self.assertNotIn("BRCA1", query)

    def test_natural_language_prompt(self):
        llm_query = 'query { gene(gene_symbol: "TP53", reference_genome: GRCh38) { symbol } }'
        llm_result = {"success": True, "data": {"query": llm_query}}
        with patch.object(database, "_query_llm_for_api", return_value=llm_result) as llm:
            result = query_gnomad(prompt="Find variants in the TP53 gene")

        self.assertEqual(llm.call_count, 1)
        self.assertTrue(result["success"])
        self.assertEqual(result["result"], self.payload)
        self.assertEqual(self.sent_query(), llm_query)


if __name__ == "__main__":
    unittest.main()
