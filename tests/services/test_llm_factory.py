import os
import sys
import sys
import unittest
from unittest.mock import patch

# Try importing project modules; if not found, add pipeline path to sys.path and retry.
try:
    from services.llm.cloud_adapter import CloudAdapter
    from services.llm.factory import LLMFactory
    from services.llm.ollama_adapter import OllamaAdapter
    from services.llm.registry import ModelRegistry, ModelSpec
except ImportError:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    PIPELINE = os.path.join(ROOT, "synthenexa_pipeline")
    if PIPELINE not in sys.path:
        sys.path.insert(0, PIPELINE)
    from services.llm.cloud_adapter import CloudAdapter
    from services.llm.factory import LLMFactory
    from services.llm.ollama_adapter import OllamaAdapter
    from services.llm.registry import ModelRegistry, ModelSpec


class LLMFactoryTests(unittest.TestCase):
    @patch("services.llm.ollama_adapter.requests.get")
    def test_factory_returns_cloud_adapter_for_cloud_model(self, get):
        get.side_effect = Exception("offline")
        adapter = LLMFactory().create(model="ollama:nemotron-3-super:cloud", base_url="http://example.test")

        self.assertIsInstance(adapter, CloudAdapter)
        self.assertEqual(adapter.model, "nemotron-3-super:cloud")
        self.assertEqual(adapter.base_url, "http://example.test")

    @patch("services.llm.ollama_adapter.requests.get")
    def test_factory_routes_cloud_models_to_cloud_adapter(self, get):
        get.side_effect = Exception("offline")
        adapter = LLMFactory().create(model="nemotron-3-super:cloud", base_url="http://example.test")

        self.assertIsInstance(adapter, CloudAdapter)
        self.assertEqual(adapter.model, "nemotron-3-super:cloud")

    def test_factory_rejects_unknown_provider(self):
        with self.assertRaises(ValueError):
            LLMFactory().create(model="qwen2.5:3b", provider="openai")

    def test_registry_returns_metadata_for_default_model(self):
        spec = ModelRegistry().get("nemotron-3-super:cloud")

        self.assertEqual(spec.provider, "ollama")
        self.assertEqual(spec.role, "primary_baseline")

    def test_registry_defaults_to_currently_available_models(self):
        registry = ModelRegistry()
        names = registry.names()

        self.assertIn("nemotron-3-super:cloud", names)
        self.assertIn("qwen3-coder-next:cloud", names)
        self.assertIn("gpt-oss:120b-cloud", names)
        self.assertIn("qwen3-coder:480b-cloud", names)

    def test_registry_allows_custom_model_names(self):
        spec = ModelRegistry().get("custom-local:1b")

        self.assertIsInstance(spec, ModelSpec)
        self.assertEqual(spec.name, "custom-local:1b")
        self.assertEqual(spec.role, "custom")


if __name__ == "__main__":
    unittest.main()
