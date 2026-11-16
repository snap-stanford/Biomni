import unittest
from unittest import mock

from biomni.tool.database import _query_llm_for_api, query_geo


class FakeResponse:
    content = '{"search_term": "BRCA1[gene]"}'


class FakeLLM:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return FakeResponse()


class DatabaseSchemaTemplateTest(unittest.TestCase):
    @mock.patch("biomni.tool.database._query_ncbi_database")
    @mock.patch("biomni.tool.database.get_llm")
    def test_query_geo_template_with_json_examples_is_rendered(self, get_llm, query_ncbi):
        llm = FakeLLM()
        get_llm.return_value = llm
        query_ncbi.return_value = {"database": "gds", "total_results": 0}

        result = query_geo(prompt="Find breast cancer RNA-seq datasets")

        self.assertEqual(result, {"database": "gds", "total_results": 0})
        query_ncbi.assert_called_once_with(
            database="gds",
            search_term="BRCA1[gene]",
            max_results=3,
        )
        system_prompt = llm.messages[0].content
        self.assertIn('For "RNA-seq data in breast cancer": {"search_term":', system_prompt)
        self.assertNotIn("{schema}", system_prompt)

    @mock.patch("biomni.tool.database.get_llm")
    def test_schema_substitution_preserves_literal_json_examples(self, get_llm):
        llm = FakeLLM()
        get_llm.return_value = llm
        template = """
        DATABASE SCHEMA:
        {schema}

        Return JSON like {"search_term": "BRCA1[gene]"}.
        Existing escaped example: {{"full_url": "https://example.test"}}.
        The response must include {search_term} as a literal reminder.
        """

        result = _query_llm_for_api(
            prompt="Find BRCA1 records",
            schema={"fields": ["gene", "clinical_significance"]},
            system_template=template,
        )

        self.assertTrue(result["success"])
        system_prompt = llm.messages[0].content
        self.assertIn('"fields": [', system_prompt)
        self.assertIn('Return JSON like {"search_term": "BRCA1[gene]"}.', system_prompt)
        self.assertIn('Existing escaped example: {"full_url": "https://example.test"}.', system_prompt)
        self.assertNotIn('{{"full_url"', system_prompt)
        self.assertIn("{search_term} as a literal reminder", system_prompt)

    @mock.patch("biomni.tool.database.get_llm")
    def test_template_without_schema_placeholder_does_not_raise(self, get_llm):
        llm = FakeLLM()
        get_llm.return_value = llm
        template = 'Return only {"search_term": "value"}.'

        result = _query_llm_for_api(
            prompt="Find a record",
            schema={"field": "value"},
            system_template=template,
        )

        self.assertTrue(result["success"])
        self.assertEqual(llm.messages[0].content, template)


if __name__ == "__main__":
    unittest.main()
