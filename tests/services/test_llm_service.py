import os
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPELINE = os.path.join(ROOT, "synthenexa_pipeline")
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)

from services.llm_service import LLMService


class LLMServiceTests(unittest.TestCase):
    @patch("services.llm_service.LLMFactory")
    def test_service_delegates_generate_to_adapter(self, factory_cls):
        adapter = Mock()
        adapter.generate.return_value = "generated"
        adapter.base_url = "http://ollama.test"
        factory_cls.return_value.create.return_value = adapter

        service = LLMService(model="ollama:cogito:3b", base_url="http://ollama.test")
        result = service.generate(
            prompt="prompt",
            system="system",
            max_retries=2,
            retry_delay=0.1,
            temperature=0.3,
        )

        self.assertEqual(service.model, "cogito:3b")
        self.assertEqual(result, "generated")
        adapter.generate.assert_called_once_with(
            prompt="prompt",
            system="system",
            max_retries=2,
            retry_delay=0.1,
            temperature=0.3,
        )

    def test_clean_output_compatibility_helper(self):
        self.assertEqual(LLMService._clean_output("```\nvalue\n```"), "value")


if __name__ == "__main__":
    unittest.main()
