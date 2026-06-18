import os
import unittest
from unittest.mock import MagicMock, patch

from minirag.config import Settings
from minirag.llm_engine import InferenceError, OpenRouterEngine


class TestOpenRouterEngine(unittest.TestCase):
    def test_init_with_explicit_api_key(self):
        engine = OpenRouterEngine(api_key="test-key")
        self.assertEqual(engine.api_key, "test-key")
        self.assertEqual(engine.model, "z-ai/glm-5.2")

    def test_init_with_env_var(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}):
            engine = OpenRouterEngine()
        self.assertEqual(engine.api_key, "env-key")

    @patch("minirag.llm_engine.get_settings")
    def test_init_missing_api_key_raises(self, mock_get_settings):
        mock_get_settings.return_value = Settings(openrouter_api_key=None)
        with self.assertRaises(RuntimeError) as ctx:
            OpenRouterEngine()
        self.assertIn("OpenRouter API key is required", str(ctx.exception))

    def test_prepare_messages_string_input(self):
        engine = OpenRouterEngine(api_key="k")
        result = engine._prepare_messages("hello", None)
        self.assertEqual(result, [{"role": "user", "content": "hello"}])

    def test_prepare_messages_list_input(self):
        engine = OpenRouterEngine(api_key="k")
        msgs = [{"role": "user", "content": "hi"}]
        result = engine._prepare_messages(msgs, None)
        self.assertEqual(result, msgs)

    def test_prepare_messages_with_last_response_and_reasoning(self):
        engine = OpenRouterEngine(api_key="k")
        last = {"content": "ok", "reasoning_details": [{"text": "r1"}]}
        result = engine._prepare_messages([{"role": "user", "content": "q"}], last)
        self.assertEqual(
            result,
            [
                {
                    "role": "assistant",
                    "content": "ok",
                    "reasoning_details": [{"text": "r1"}],
                },
                {"role": "user", "content": "q"},
            ],
        )

    def test_prepare_messages_with_last_response_no_reasoning(self):
        engine = OpenRouterEngine(api_key="k")
        last = {"content": "ok"}
        result = engine._prepare_messages([{"role": "user", "content": "q"}], last)
        self.assertEqual(
            result,
            [
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "q"},
            ],
        )

    @patch("minirag.llm_engine.requests.post")
    def test_generate_success(self, mock_post):
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"choices": [{"message": {"content": "hi"}}]}),
        )
        engine = OpenRouterEngine(model="m", api_key="k")
        result = engine.generate("hello")

        self.assertEqual(result, {"content": "hi"})
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer k")
        payload = kwargs["json"]
        self.assertEqual(payload["model"], "m")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])
        self.assertTrue(payload["reasoning"]["enabled"])

    @patch("minirag.llm_engine.requests.post")
    def test_generate_request_exception(self, mock_post):
        import requests

        mock_post.side_effect = requests.ConnectionError("boom")
        engine = OpenRouterEngine(api_key="k")
        with self.assertRaises(InferenceError) as ctx:
            engine.generate("hello")
        self.assertIn("LLM inference failed", str(ctx.exception))

    @patch("minirag.llm_engine.requests.post")
    def test_generate_http_error(self, mock_post):
        import requests

        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(side_effect=requests.HTTPError("403")),
        )
        engine = OpenRouterEngine(api_key="k")
        with self.assertRaises(InferenceError) as ctx:
            engine.generate("hello")
        self.assertIn("LLM inference failed", str(ctx.exception))

    @patch("minirag.llm_engine.requests.post")
    def test_generate_keyerror_response(self, mock_post):
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={}),
        )
        engine = OpenRouterEngine(api_key="k")
        with self.assertRaises(InferenceError) as ctx:
            engine.generate("hello")
        self.assertIn("Unexpected response format", str(ctx.exception))

    @patch("minirag.llm_engine.requests.post")
    def test_generate_indexerror_response(self, mock_post):
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"choices": []}),
        )
        engine = OpenRouterEngine(api_key="k")
        with self.assertRaises(InferenceError) as ctx:
            engine.generate("hello")
        self.assertIn("Unexpected response format", str(ctx.exception))

    @patch("minirag.llm_engine.requests.post")
    def test_generate_typeerror_response(self, mock_post):
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"choices": "bad"}),
        )
        engine = OpenRouterEngine(api_key="k")
        with self.assertRaises(InferenceError) as ctx:
            engine.generate("hello")
        self.assertIn("Unexpected response format", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
