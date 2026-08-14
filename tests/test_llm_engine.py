import unittest
from unittest.mock import MagicMock, patch

from minirag.adapters.llm import InferenceError, OpenRouterEngine


class TestOpenRouterEngine(unittest.TestCase):
    def test_init_with_explicit_api_key_and_model(self):
        engine = OpenRouterEngine(model="z-ai/glm-5.2", api_key="test-key")
        self.assertEqual(engine.api_key, "test-key")
        self.assertEqual(engine.model, "z-ai/glm-5.2")

    def test_init_missing_model_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            OpenRouterEngine(model="", api_key="test-key")
        self.assertIn("OpenRouter API key is required", str(ctx.exception))

    def test_init_missing_api_key_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            OpenRouterEngine(model="z-ai/glm-5.2", api_key=None)
        self.assertIn("OpenRouter API key is required", str(ctx.exception))

    def test_init_rejects_non_positive_timeout(self):
        with self.assertRaisesRegex(ValueError, "timeout"):
            OpenRouterEngine(model="m", api_key="k", timeout=0)

    def test_prepare_messages_string_input(self):
        engine = OpenRouterEngine(model="m", api_key="k")
        result = engine._prepare_messages("hello", None)
        self.assertEqual(result, [{"role": "user", "content": "hello"}])

    def test_prepare_messages_list_input(self):
        engine = OpenRouterEngine(model="m", api_key="k")
        msgs = [{"role": "user", "content": "hi"}]
        result = engine._prepare_messages(msgs, None)
        self.assertEqual(result, msgs)
        self.assertIsNot(result, msgs)
        self.assertIsNot(result[0], msgs[0])
        result[0]["content"] = "changed"
        self.assertEqual(msgs[0]["content"], "hi")

    def test_prepare_messages_with_last_response_and_reasoning(self):
        engine = OpenRouterEngine(model="m", api_key="k")
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
        engine = OpenRouterEngine(model="m", api_key="k")
        last = {"content": "ok"}
        result = engine._prepare_messages([{"role": "user", "content": "q"}], last)
        self.assertEqual(
            result,
            [
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "q"},
            ],
        )

    @patch("minirag.adapters.llm.get_client")
    @patch("minirag.adapters.llm.requests.post")
    def test_generate_success(self, mock_post, mock_get_client):
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(
                return_value={
                    "choices": [{"message": {"content": "hi"}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                }
            ),
        )
        engine = OpenRouterEngine(model="m", api_key="k", timeout=12)
        result = engine.generate(
            "hello",
            reasoning=False,
            schema={},
            span_name="test-generation",
        )

        self.assertEqual(result, {"content": "hi"})
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer k")
        payload = kwargs["json"]
        self.assertEqual(payload["model"], "m")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])
        self.assertFalse(payload["reasoning"]["enabled"])
        self.assertEqual(
            payload["response_format"],
            {
                "type": "json_schema",
                "json_schema": {"name": "miniRAG", "strict": True, "schema": {}},
            },
        )
        self.assertEqual(kwargs["timeout"], 12)
        mock_get_client.return_value.update_current_generation.assert_called_once_with(
            name="test-generation",
            model="m",
            input=[{"role": "user", "content": "hello"}],
            output={"content": "hi"},
            usage_details={"input": 7, "output": 3},
        )

    @patch("minirag.adapters.llm.get_client")
    @patch("minirag.adapters.llm.requests.post")
    def test_generate_tolerates_malformed_usage(self, mock_post, mock_get_client):
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": "unknown", "completion_tokens": -1},
        }
        engine = OpenRouterEngine(model="m", api_key="k")

        self.assertEqual(engine.generate("hello"), {"content": "hi"})

        call_kwargs = (
            mock_get_client.return_value.update_current_generation.call_args.kwargs
        )
        self.assertIsNone(call_kwargs["usage_details"])

    @patch("minirag.adapters.llm.requests.post")
    def test_generate_request_exception(self, mock_post):
        import requests

        mock_post.side_effect = requests.ConnectionError("boom")
        engine = OpenRouterEngine(model="m", api_key="k")
        with self.assertRaises(InferenceError) as ctx:
            engine.generate("hello")
        self.assertIn("LLM inference failed", str(ctx.exception))

    @patch("minirag.adapters.llm.requests.post")
    def test_generate_http_error(self, mock_post):
        import requests

        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(side_effect=requests.HTTPError("403")),
        )
        engine = OpenRouterEngine(model="m", api_key="k")
        with self.assertRaises(InferenceError) as ctx:
            engine.generate("hello")
        self.assertIn("LLM inference failed", str(ctx.exception))

    @patch("minirag.adapters.llm.requests.post")
    def test_generate_invalid_json(self, mock_post):
        mock_post.return_value = MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(side_effect=ValueError("bad json")),
        )
        engine = OpenRouterEngine(model="m", api_key="k")

        with self.assertRaisesRegex(InferenceError, "invalid JSON"):
            engine.generate("hello")

    @patch("minirag.adapters.llm.requests.post")
    def test_generate_rejects_invalid_core_response(self, mock_post):
        engine = OpenRouterEngine(model="m", api_key="k")
        cases = [
            ([], "expected an object"),
            ({"error": {"message": "bad request"}}, "OpenRouter API error"),
            (
                {"choices": []},
                "missing assistant message",
            ),
            (
                {"choices": [{"message": "bad"}]},
                "message must be an object",
            ),
        ]

        for payload, message in cases:
            with (
                self.subTest(payload=payload),
                self.assertRaisesRegex(InferenceError, message),
            ):
                mock_post.return_value.json.return_value = payload
                engine.generate("hello")


if __name__ == "__main__":
    unittest.main()
