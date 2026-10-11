import io
import os
import unittest
from contextlib import redirect_stdout
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from biomni.agent import a1
from biomni.config import BiomniConfig


class ConfigApiKeyLoggingTest(unittest.TestCase):
    def test_initialization_masks_keys_without_changing_model_credentials(self):
        for key in ("biomni-dummy-secret-1234", "shortkey", "", None):
            with self.subTest(key=key), TemporaryDirectory() as path, patch.dict(os.environ, {}, clear=True):
                config = BiomniConfig(path=path, api_key=key, source="Custom", use_tool_retriever=False)
                output = io.StringIO()
                override_key = "biomni-dummy-override-5678"
                with (
                    patch.object(a1, "default_config", config),
                    patch.object(a1, "read_module2api", return_value={}),
                    patch.object(a1, "get_llm") as get_llm,
                    patch.object(a1, "KnowHowLoader", return_value=SimpleNamespace(documents=[])),
                    patch.object(a1.A1, "configure"),
                    redirect_stdout(output),
                ):
                    a1.A1(expected_data_lake_files=[])
                    self.assertEqual(get_llm.call_args.kwargs["api_key"], key or "EMPTY")
                    a1.A1(llm="override-model", api_key=override_key, expected_data_lake_files=[])
                    self.assertEqual(get_llm.call_args.kwargs["api_key"], override_key)

                log = output.getvalue()
                if key:
                    self.assertNotIn(key, log)
                    masked = "********1234" if len(key) > 8 else "***"
                    self.assertIn(f"Api Key: {masked}\n", log)
                self.assertNotIn(override_key, log)
                self.assertIn("API Key: ********5678\n", log)


if __name__ == "__main__":
    unittest.main()
