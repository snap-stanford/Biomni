import inspect
import unittest
from unittest import mock

from biomni.tool.database import query_civic
from biomni.tool.tool_description.database import description


def graphql_response(data):
    return {"success": True, "result": {"data": data}}


def variant_response(total_count=1, nodes=None):
    if nodes is None:
        nodes = [{"id": 79, "name": "G12D", "link": "/variants/79", "variantAliases": ["GLY12ASP"]}]
    return graphql_response(
        {
            "gene": {
                "id": 30,
                "name": "KRAS",
                "entrezId": 3845,
                "link": "/features/30",
                "variants": {"totalCount": total_count, "nodes": nodes},
            }
        }
    )


def evidence_item(item_id=1300, rating=4):
    return {
        "id": item_id,
        "name": f"EID{item_id}",
        "link": f"/evidence/{item_id}",
        "status": "ACCEPTED",
        "description": "Curated variant evidence.",
        "evidenceType": "PROGNOSTIC",
        "evidenceLevel": "B",
        "evidenceRating": rating,
        "evidenceDirection": "SUPPORTS",
        "significance": "POOR_OUTCOME",
        "variantOrigin": "SOMATIC",
        "disease": {"id": 556, "doid": "1793", "name": "Pancreatic Cancer", "displayName": "Pancreatic Cancer"},
        "therapies": [],
        "source": {
            "sourceType": "PUBMED",
            "citationId": "27010960",
            "pmcId": "PMC4822095",
            "title": "KRAS G12D Mutation Subtype Is A Prognostic Factor",
        },
    }


def evidence_response(total_count=1, nodes=None):
    if nodes is None:
        nodes = [evidence_item()]
    return graphql_response({"evidenceItems": {"totalCount": total_count, "nodes": nodes}})


class CivicQueryTest(unittest.TestCase):
    @mock.patch("biomni.tool.database._query_rest_api")
    def test_returns_normalized_source_linked_evidence(self, request):
        request.side_effect = [variant_response(), evidence_response()]

        result = query_civic(
            " kras ",
            "KRAS p.G12D",
            disease=" pancreatic ",
            evidence_type="prognostic",
            max_results=2,
        )

        self.assertEqual(result["query"]["gene"], "KRAS")
        self.assertEqual(result["query"]["variant"], "G12D")
        self.assertEqual(result["query"]["disease"], "pancreatic")
        self.assertEqual(result["query"]["evidence_type"], "PROGNOSTIC")
        self.assertEqual(result["query"]["evidence_status"], "ACCEPTED")
        self.assertEqual(result["gene"]["url"], "https://civicdb.org/features/30")
        self.assertEqual(result["matched_variants"][0]["url"], "https://civicdb.org/variants/79")
        self.assertEqual(result["returned_evidence"], 1)
        item = result["evidence_items"][0]
        self.assertEqual(item["url"], "https://civicdb.org/evidence/1300")
        self.assertEqual(item["variant"]["name"], "G12D")
        self.assertEqual(item["source"]["pubmed_url"], "https://pubmed.ncbi.nlm.nih.gov/27010960/")
        self.assertEqual(item["source"]["pmc_url"], "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4822095/")
        self.assertIn("not clinical recommendations", result["notes"][1])

        variant_call, evidence_call = request.call_args_list
        self.assertEqual(
            variant_call.kwargs["json_data"]["variables"],
            {"gene": "KRAS", "variant": "G12D", "first": 5},
        )
        self.assertEqual(
            evidence_call.kwargs["json_data"]["variables"],
            {
                "variantId": 79,
                "disease": "pancreatic",
                "evidenceType": "PROGNOSTIC",
                "status": "ACCEPTED",
                "first": 2,
            },
        )

    @mock.patch("biomni.tool.database._query_rest_api")
    def test_caps_alias_matches_and_total_evidence(self, request):
        nodes = [
            {"id": 148, "name": "G12A", "link": "/variants/148", "variantAliases": ["RS121913529"]},
            {"id": 79, "name": "G12D", "link": "/variants/79", "variantAliases": ["RS121913529"]},
        ]
        request.side_effect = [
            variant_response(total_count=3, nodes=nodes),
            evidence_response(total_count=4, nodes=[evidence_item(1)]),
            evidence_response(total_count=2, nodes=[evidence_item(2)]),
        ]

        result = query_civic("KRAS", "RS121913529", max_results=2, max_variants=2)

        self.assertEqual(result["total_matching_variants"], 3)
        self.assertEqual(result["total_evidence_for_returned_variants"], 6)
        self.assertEqual(result["returned_evidence"], 2)
        self.assertEqual([item["variant"]["name"] for item in result["evidence_items"]], ["G12A", "G12D"])
        self.assertTrue(any("only the first 2" in note for note in result["notes"]))
        self.assertTrue(any("max_results returned 2" in note for note in result["notes"]))

    @mock.patch("biomni.tool.database._query_rest_api")
    def test_missing_gene_and_variant_return_structured_empty_results(self, request):
        request.side_effect = [
            graphql_response({"gene": None}),
            variant_response(total_count=0, nodes=[]),
        ]

        missing_gene = query_civic("NOTAGENE", "V1")
        missing_variant = query_civic("KRAS", "NOTAVARIANT")

        self.assertIn("was not found", missing_gene["error"])
        self.assertEqual(missing_gene["evidence_items"], [])
        self.assertEqual(missing_variant["total_matching_variants"], 0)
        self.assertEqual(missing_variant["returned_evidence"], 0)
        self.assertIn("No CIViC variant", missing_variant["note"])

    @mock.patch("biomni.tool.database._query_rest_api")
    def test_surfaces_graphql_and_transport_errors(self, request):
        request.side_effect = [
            {"success": True, "result": {"errors": [{"message": "invalid query"}]}},
            {"success": False, "error": "API error: offline"},
        ]

        graphql_error = query_civic("KRAS", "G12D")
        transport_error = query_civic("KRAS", "G12D")

        self.assertEqual(graphql_error["error"], "CIViC GraphQL error: invalid query")
        self.assertEqual(transport_error["error"], "API error: offline")

    @mock.patch("biomni.tool.database._query_rest_api")
    def test_nonaccepted_status_is_explicitly_warned(self, request):
        submitted = evidence_item()
        submitted["status"] = "SUBMITTED"
        request.side_effect = [variant_response(), evidence_response(nodes=[submitted])]

        result = query_civic("KRAS", "G12D", evidence_status="submitted")

        self.assertEqual(result["query"]["evidence_status"], "SUBMITTED")
        self.assertTrue(any("unreviewed or rejected" in note for note in result["notes"]))
        evidence_payload = request.call_args_list[1].kwargs["json_data"]
        self.assertNotIn("disease", evidence_payload["variables"])
        self.assertNotIn("evidenceType", evidence_payload["variables"])
        self.assertNotIn("diseaseName: $disease", evidence_payload["query"])
        self.assertNotIn("evidenceType: $evidenceType", evidence_payload["query"])

    @mock.patch("biomni.tool.database._query_rest_api")
    def test_validates_inputs_before_network_calls(self, request):
        cases = [
            (("", "G12D"), {}, "gene"),
            (("KRAS", ""), {}, "variant"),
            (("KRAS", "G12D"), {"disease": ""}, "disease"),
            (("KRAS", "G12D"), {"evidence_type": "unknown"}, "evidence_type"),
            (("KRAS", "G12D"), {"evidence_status": "pending"}, "evidence_status"),
            (("KRAS", "G12D"), {"max_results": 0}, "max_results"),
            (("KRAS", "G12D"), {"max_results": True}, "max_results"),
            (("KRAS", "G12D"), {"max_variants": 21}, "max_variants"),
        ]
        for args, kwargs, expected in cases:
            with self.subTest(args=args, kwargs=kwargs):
                self.assertIn(expected, query_civic(*args, **kwargs)["error"])
        request.assert_not_called()

    def test_tool_description_matches_signature(self):
        tool = next(item for item in description if item["name"] == "query_civic")
        schema_required = {item["name"] for item in tool["required_parameters"]}
        schema_defaults = {item["name"]: item["default"] for item in tool["optional_parameters"]}
        signature = inspect.signature(query_civic)
        function_required = {
            name for name, parameter in signature.parameters.items() if parameter.default is inspect.Parameter.empty
        }
        function_defaults = {
            name: parameter.default
            for name, parameter in signature.parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }

        self.assertEqual(schema_required, function_required)
        self.assertEqual(schema_defaults, function_defaults)


if __name__ == "__main__":
    unittest.main()
