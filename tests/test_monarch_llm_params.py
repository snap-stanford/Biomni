"""Offline checks that query_monarch sends the query parameters the model returns."""

import json
import unittest
from unittest.mock import patch

import requests
from biomni.tool.database import query_monarch

BASE = "https://api.monarchinitiative.org/v3/api"


class MonarchLLMParamsTest(unittest.TestCase):
    def setUp(self):
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"total": 0, "items": []}).encode()
        request_patch = patch("requests.sessions.Session.request", return_value=response)
        self.request = request_patch.start()
        self.addCleanup(request_patch.stop)

    def query_with_llm_output(self, data, **kwargs):
        llm_result = {"success": True, "data": data, "raw_response": json.dumps(data)}
        with patch("biomni.tool.database._query_llm_for_api", return_value=llm_result):
            result = query_monarch(prompt="Find the disease entity for Marfan syndrome", verbose=True, **kwargs)
        self.assertTrue(result["success"])
        self.assertEqual(self.request.call_count, 1)
        return self.request.call_args.kwargs["url"], self.request.call_args.kwargs.get("params") or {}

    def test_search_params_from_model_are_sent(self):
        url, params = self.query_with_llm_output(
            {
                "endpoint": "search",
                "url": f"{BASE}/search",
                "params": {"q": "marfan", "category": "biolink:Disease"},
            }
        )
        self.assertEqual(url, f"{BASE}/search?limit=2")
        self.assertEqual(params, {"q": "marfan", "category": "biolink:Disease"})

    def test_limit_from_model_is_not_sent_twice(self):
        url, params = self.query_with_llm_output(
            {"endpoint": "search", "url": f"{BASE}/search", "params": {"q": "marfan", "limit": 10}}
        )
        self.assertEqual(url, f"{BASE}/search")
        self.assertEqual(params, {"q": "marfan", "limit": 10})

    def test_model_output_without_params(self):
        url, params = self.query_with_llm_output({"endpoint": "get_entity", "url": f"{BASE}/entity/MONDO:0007947"})
        self.assertEqual(url, f"{BASE}/entity/MONDO:0007947?limit=2")
        self.assertEqual(params, {})

    def test_direct_endpoint_is_unchanged(self):
        result = query_monarch(endpoint="/entity/MONDO:0007947", verbose=True)
        self.assertTrue(result["success"])
        self.assertEqual(self.request.call_args.kwargs["url"], f"{BASE}/entity/MONDO:0007947?limit=2")
        self.assertFalse(self.request.call_args.kwargs.get("params"))


if __name__ == "__main__":
    unittest.main()
