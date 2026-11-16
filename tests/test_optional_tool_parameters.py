import unittest

from biomni.tool.tool_description.glycoengineering import description
from biomni.utils import api_schema_to_langchain_tool
from pydantic import ValidationError


def append_to_labels(value, labels=None):
    if labels is None:
        labels = []
    labels.append(value)
    return labels


class OptionalToolParametersTest(unittest.TestCase):
    def test_existing_optional_parameter_is_visible_and_forwarded(self):
        schema = next(item for item in description if item["name"] == "find_n_glycosylation_motifs")
        tool = api_schema_to_langchain_tool(
            schema,
            mode="custom_tool",
            module_name="biomni.tool.glycoengineering",
        )

        properties = tool.args_schema.model_json_schema()["properties"]
        self.assertEqual(properties["allow_overlap"]["default"], False)
        self.assertEqual(
            properties["allow_overlap"]["description"],
            "Allow overlapping motif detections",
        )
        self.assertNotIn("allow_overlap", tool.args_schema.model_json_schema()["required"])

        default_result = tool.invoke({"sequence": "NNST"})
        override_result = tool.invoke({"sequence": "NNST", "allow_overlap": True})

        self.assertIn("Total sequons found: 1", default_result)
        self.assertIn("Total sequons found: 2", override_result)
        self.assertIn("- 2: NST", override_result)

    def test_required_parameter_validation_is_unchanged(self):
        schema = next(item for item in description if item["name"] == "find_n_glycosylation_motifs")
        tool = api_schema_to_langchain_tool(
            schema,
            mode="custom_tool",
            module_name="biomni.tool.glycoengineering",
        )

        with self.assertRaises(ValidationError):
            tool.invoke({"allow_overlap": True})

    def test_optional_list_values_are_forwarded_without_leaking_between_calls(self):
        schema = {
            "name": "append_to_labels",
            "description": "Append one value to a caller-specific label list.",
            "required_parameters": [
                {
                    "name": "value",
                    "type": "str",
                    "description": "Value to append",
                    "default": None,
                }
            ],
            "optional_parameters": [
                {
                    "name": "labels",
                    "type": "List[str]",
                    "description": "Initial labels",
                    "default": None,
                }
            ],
        }
        tool = api_schema_to_langchain_tool(schema, mode="custom_tool", module_name=__name__)

        self.assertEqual(tool.invoke({"value": "first"}), ["first"])
        self.assertEqual(tool.invoke({"value": "second"}), ["second"])
        self.assertEqual(tool.invoke({"value": "last", "labels": ["initial"]}), ["initial", "last"])

    def test_function_signature_corrects_stale_description_defaults(self):
        from biomni.tool.tool_description.bioengineering import description as bioengineering_description

        schema = next(item for item in bioengineering_description if item["name"] == "simulate_whole_cell_ode_model")
        tool = api_schema_to_langchain_tool(
            schema,
            mode="custom_tool",
            module_name="biomni.tool.bioengineering",
        )
        properties = tool.args_schema.model_json_schema()["properties"]

        self.assertEqual(properties["time_span"]["default"], [0, 100])
        self.assertEqual(properties["method"]["default"], "LSODA")


if __name__ == "__main__":
    unittest.main()
