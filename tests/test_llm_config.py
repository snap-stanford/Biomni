import os
import unittest
from unittest.mock import patch

from biomni.config import BiomniConfig
from biomni.llm import get_llm
from langchain_openai import ChatOpenAI


class LLMConfigTest(unittest.TestCase):
    def test_config_model_and_explicit_override(self):
        with patch.dict(os.environ, {}, clear=True):
            config = BiomniConfig(
                llm="configured-model",
                source="Custom",
                base_url="http://127.0.0.1:8000/v1",
                api_key="test-key",
                temperature=0.2,
            )
            for model, expected in [(None, "configured-model"), ("explicit-model", "explicit-model")]:
                with self.subTest(model=model):
                    llm = get_llm(model=model, config=config)
                    self.assertIsInstance(llm, ChatOpenAI)
                    self.assertEqual(llm.model_name, expected)
                    self.assertEqual(llm.temperature, config.temperature)
                    self.assertEqual(llm.openai_api_base, config.base_url)


if __name__ == "__main__":
    unittest.main()
