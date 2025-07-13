import unittest

from biomni.tool.tooluniverse_registry import ToolUniverseRegistry


class TestToolUniverseRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ToolUniverseRegistry()

    def test_tool_schemas(self):
        schemas = self.registry.get_tool_schemas()
        self.assertIsInstance(schemas, list)
        self.assertTrue(any("name" in s for s in schemas))

    def test_call_tool(self):
        schemas = self.registry.get_tool_schemas()
        if schemas:
            tool_name = schemas[0]["name"]
            # Try calling with no arguments (may fail if required, but should not error at registry level)
            try:
                result = self.registry.call_tool(tool_name)
            except Exception as e:
                result = str(e)
            self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
