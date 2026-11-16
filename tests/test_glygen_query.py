import inspect
import unittest
from unittest import mock

import requests
from biomni.tool.database import query_glygen
from biomni.tool.tool_description.database import description


def response_with(payload):
    response = mock.MagicMock()
    response.json.return_value = payload
    return response


def protein_payload():
    return {
        "uniprot": {"uniprot_canonical_ac": "P00533-1"},
        "protein_names": [
            {"name": "EGFR synonym", "type": "synonym"},
            {"name": "Epidermal growth factor receptor", "type": "recommended"},
        ],
        "gene_names": [{"name": "EGFR"}, {"name": "ERBB1"}, {"name": "EGFR"}],
        "species": [{"name": "Homo sapiens", "taxid": 9606}],
        "mass": 134277,
        "sequence": {"sequence": "MNST", "length": 4},
        "glycosylation": [
            {
                "start_pos": 56,
                "end_pos": 56,
                "residue": "Asn",
                "site_lbl": "Asn56",
                "site_seq": "VCQGTSNKLTQ",
                "type": "N-linked",
                "subtype": "other",
                "site_category": "reported_with_glycan",
                "glytoucan_ac": "G11111AA",
                "evidence": [
                    {"database": "PubMed", "id": "1", "url": "https://pubmed.example/1"},
                    {"database": "PubMed", "id": "2", "url": "https://pubmed.example/2"},
                ],
            },
            {
                "start_pos": 128,
                "end_pos": 128,
                "residue": "Asn",
                "type": "N-linked",
                "site_category": "reported",
                "evidence": [],
            },
            {"start_pos": 1096, "end_pos": 1096, "residue": "Ser", "type": "O-linked"},
        ],
    }


class GlyGenQueryTest(unittest.TestCase):
    @mock.patch("biomni.tool.database.requests.post")
    def test_protein_record_filters_sites_and_preserves_sources(self, post):
        post.return_value = response_with(protein_payload())

        result = query_glygen(
            " p00533 ",
            record_type="uniprot",
            glycosylation_type="n-LINKED",
            max_results=1,
            max_evidence=1,
            include_sequence=True,
            timeout=12,
        )

        post.assert_called_once_with(
            "https://api.glygen.org/protein/detail/P00533/",
            json={},
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=12,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["protein"]["canonical_uniprot_accession"], "P00533-1")
        self.assertEqual(result["protein"]["recommended_name"], "Epidermal growth factor receptor")
        self.assertEqual(result["protein"]["gene_names"], ["EGFR", "ERBB1"])
        self.assertEqual(result["protein"]["sequence"], "MNST")
        self.assertEqual(result["matching_glycosylation_associations"], 2)
        self.assertEqual(result["unique_matching_sites"], 2)
        self.assertEqual(result["returned_associations"], 1)
        site = result["glycosylation_sites"][0]
        self.assertEqual(site["glycan_url"], "https://glygen.org/glycan/G11111AA")
        self.assertEqual(site["evidence_count"], 2)
        self.assertEqual(len(site["evidence"]), 1)
        self.assertTrue(any("first 1 of 2" in note for note in result["notes"]))

    @mock.patch("biomni.tool.database.requests.post")
    def test_glycan_record_normalizes_associations_and_reports_truncation(self, post):
        post.return_value = response_with(
            {
                "glytoucan": {
                    "glytoucan_ac": "G17689DH",
                    "glytoucan_url": "https://glytoucan.org/Structures/Glycans/G17689DH",
                },
                "mass": 2368.84,
                "number_monosaccharides": 12,
                "glycan_type": "Saccharide",
                "iupac": "full IUPAC",
                "iupac_condensed": "condensed IUPAC",
                "wurcs": "WURCS=2.0/example",
                "composition": [{"name": "Hexose", "count": 5}],
                "classification": [{"type": {"name": "N-linked"}, "subtype": {"name": "Complex"}}],
                "species": [
                    {
                        "name": "Homo sapiens",
                        "common_name": "Human",
                        "taxid": 9606,
                        "evidence": [{"database": "PubMed", "id": "10", "url": "https://example/10"}],
                    },
                    {"name": "Mus musculus", "taxid": 10090},
                ],
                "glycoprotein": [
                    {
                        "uniprot_canonical_ac": "P01588-1",
                        "protein_name": "Erythropoietin",
                        "gene_name": "EPO",
                        "start_pos": 65,
                        "residue": "asn",
                        "tax_id": 9606,
                        "tax_name": "Homo sapiens",
                        "evidence": [],
                    }
                ],
                "enzyme": [],
                "publication": [
                    {
                        "title": "A glycan paper",
                        "journal": "Glycobiology",
                        "date": "2025",
                        "authors": "A. Author",
                        "reference": [
                            {
                                "type": "PubMed",
                                "id": "123",
                                "url": "https://glygen.org/publication/PubMed/123",
                            }
                        ],
                    }
                ],
            }
        )

        result = query_glygen("g17689dh", max_results=1)

        post.assert_called_once_with(
            "https://api.glygen.org/glycan/detail/G17689DH/",
            json={},
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=30,
        )
        self.assertEqual(result["glycan"]["classification"], [{"type": "N-linked", "subtype": "Complex"}])
        self.assertEqual(result["counts"], {"species": 2, "glycoproteins": 1, "enzymes": 0, "publications": 1})
        self.assertEqual(result["species"][0]["taxid"], 9606)
        self.assertEqual(result["glycoproteins"][0]["gene_name"], "EPO")
        self.assertEqual(result["publications"][0]["references"][0]["database"], "PubMed")
        self.assertTrue(any("truncated sections: species" in note for note in result["notes"]))

    @mock.patch("biomni.tool.database.requests.post")
    def test_zero_evidence_limit_omits_source_links(self, post):
        post.return_value = response_with(protein_payload())

        result = query_glygen("P00533", max_evidence=0)

        self.assertEqual(result["glycosylation_sites"][0]["evidence_count"], 2)
        self.assertEqual(result["glycosylation_sites"][0]["evidence"], [])

    @mock.patch("biomni.tool.database.requests.post")
    def test_validates_inputs_before_network_calls(self, post):
        cases = [
            (("",), {}, "identifier"),
            (("P00533",), {"record_type": "unknown"}, "record_type"),
            (("../secret",), {}, "UniProt"),
            (("P00533",), {"max_results": 0}, "max_results"),
            (("P00533",), {"max_results": True}, "max_results"),
            (("P00533",), {"max_evidence": 11}, "max_evidence"),
            (("P00533",), {"timeout": 0}, "timeout"),
            (("P00533",), {"include_sequence": "yes"}, "include_sequence"),
            (("G17689DH",), {"record_type": "glycan", "glycosylation_type": "N-linked"}, "protein"),
            (("not-a-glycan",), {"record_type": "glycan"}, "GlyTouCan"),
        ]
        for args, kwargs, expected in cases:
            with self.subTest(args=args, kwargs=kwargs):
                self.assertIn(expected, query_glygen(*args, **kwargs)["error"])
        post.assert_not_called()

    @mock.patch("biomni.tool.database.requests.post")
    def test_surfaces_transport_and_response_errors(self, post):
        post.side_effect = requests.Timeout("offline")
        self.assertIn("request failed", query_glygen("P00533")["error"])

        post.side_effect = None
        post.return_value = response_with(["unexpected"])
        self.assertIn("unexpected response", query_glygen("P00533")["error"])

        post.return_value = response_with({"error": "record not found"})
        self.assertEqual(query_glygen("P00533")["error"], "GlyGen API error: record not found")

    def test_tool_description_matches_signature(self):
        tool = next(item for item in description if item["name"] == "query_glygen")
        schema_required = {item["name"] for item in tool["required_parameters"]}
        schema_defaults = {item["name"]: item["default"] for item in tool["optional_parameters"]}
        signature = inspect.signature(query_glygen)
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
