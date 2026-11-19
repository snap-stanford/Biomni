import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock

from biomni.utils import function_to_api_schema


class SchemaRetryTest(unittest.TestCase):
    def test_transient_failure_is_retried(self):
        llm = Mock()
        llm.with_structured_output.return_value = llm
        llm.invoke.side_effect = [
            RuntimeError("temporary provider failure"),
            SimpleNamespace(dict=lambda: {"api_schema": "{'name': 'demo'}"}),
        ]
        with redirect_stdout(io.StringIO()):
            result = function_to_api_schema("def demo(): pass", llm)
        self.assertEqual(result, {"name": "demo"})
        self.assertEqual(llm.invoke.call_count, 2)

    def test_repeated_failures_return_existing_error(self):
        llm = Mock()
        llm.with_structured_output.return_value = llm
        llm.invoke.side_effect = RuntimeError("provider unavailable")
        with redirect_stdout(io.StringIO()):
            result = function_to_api_schema("def demo(): pass", llm)
        self.assertEqual(result, "Error: Could not parse the API schema")
        self.assertEqual(llm.invoke.call_count, 7)


if __name__ == "__main__":
    unittest.main()
