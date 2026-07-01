import importlib
import os
import sys
import unittest
from unittest.mock import Mock, patch

# Ensure local pipeline package is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPELINE = os.path.join(ROOT, "synthenexa_pipeline")
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)

OllamaAdapter = importlib.import_module("services.llm.ollama_adapter").OllamaAdapter


class OllamaAdapterTests(unittest.TestCase):
    def test_clean_output_removes_markdown_fence(self):
        text = "```python\ndef add(a, b):\n    return a + b\n```"

        self.assertEqual(
            OllamaAdapter.clean_output(text),
            "def add(a, b):\n    return a + b",
        )

    @patch("services.llm.ollama_adapter.requests.post")
    def test_generate_formats_chat_payload(self, post):
        response = Mock()
        response.json.return_value = {"message": {"content": "hello"}}
        response.raise_for_status.return_value = None
        post.return_value = response
        adapter = OllamaAdapter(
            model="ollama:nemotron-3-super:cloud",
            base_url="http://ollama.test",
            temperature=0.4,
            timeout_seconds=12,
            max_retries=1,
        )

        result = adapter.generate(prompt="Say hello", system="Be brief", temperature=0.2)

        self.assertEqual(result, "hello")
        post.assert_called_once()
        url = post.call_args.args[0]
        payload = post.call_args.kwargs["json"]
        timeout = post.call_args.kwargs["timeout"]
        self.assertEqual(url, "http://ollama.test/api/chat")
        self.assertEqual(payload["model"], "nemotron-3-super:cloud")
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "Be brief"})
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "Say hello"})
        self.assertEqual(payload["options"]["temperature"], 0.2)
        self.assertEqual(timeout, 12)

    @patch("services.llm.ollama_adapter.time.sleep")
    @patch("services.llm.ollama_adapter.requests.post")
    def test_generate_retries_failed_requests(self, post, sleep):
        response = Mock()
        response.json.return_value = {"message": {"content": "ok"}}
        response.raise_for_status.return_value = None
        post.side_effect = [RuntimeError("temporary"), response]
        adapter = OllamaAdapter(
            model="qwen2.5:3b",
            base_url="http://ollama.test",
            timeout_seconds=12,
            max_retries=2,
        )

        result = adapter.generate(prompt="Hello")

        self.assertEqual(result, "ok")
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
