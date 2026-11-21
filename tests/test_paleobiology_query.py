"""Offline checks that query_paleobiology loads its bundled schema and builds valid PBDB URLs."""

import json
import unittest
from unittest.mock import patch

import requests
from biomni.tool import database
from biomni.tool.database import query_paleobiology


class PaleobiologyQueryTest(unittest.TestCase):
    def setUp(self):
        self.payload = {"records": [{"nam": "Tyrannosaurus"}]}
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(self.payload).encode()
        request_patch = patch("requests.sessions.Session.request", return_value=response)
        self.request = request_patch.start()
        self.addCleanup(request_patch.stop)

    def test_prompt_path_loads_the_bundled_schema(self):
        url = "https://paleobiodb.org/data1.2/taxa/list.json?name=Tyrannosaurus"
        llm_result = {"success": True, "data": {"full_url": url, "description": "Tyrannosaurus taxa"}}
        with patch.object(database, "_query_llm_for_api", return_value=llm_result) as llm:
            result = query_paleobiology(prompt="Find fossil records of Tyrannosaurus rex")

        schema = llm.call_args.kwargs["schema"]
        self.assertEqual(schema["base_url"], "https://paleobiodb.org/data1.2")
        self.assertTrue(result["success"])
        self.assertEqual(result["result"], self.payload)
        self.assertEqual(self.request.call_args.kwargs["url"], url)

    def test_documented_direct_endpoint_is_not_doubled(self):
        result = query_paleobiology(endpoint="taxa/list.json?name=Tyrannosaurus")

        self.assertTrue(result["success"])
        self.assertEqual(
            self.request.call_args.kwargs["url"],
            "https://paleobiodb.org/data1.2/taxa/list.json?name=Tyrannosaurus",
        )


if __name__ == "__main__":
    unittest.main()
