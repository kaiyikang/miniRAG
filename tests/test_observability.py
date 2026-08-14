import os
from unittest.mock import patch

from minirag.observability import _configure_langfuse, get_client, observe


def test_disables_tracing_when_credentials_are_missing():
    with patch.dict(
        os.environ,
        {"LANGFUSE_TRACING_ENABLED": "true"},
        clear=True,
    ):
        configured = _configure_langfuse()

        assert configured is False
        assert os.environ["LANGFUSE_TRACING_ENABLED"] == "false"


def test_preserves_tracing_setting_when_credentials_are_complete():
    with patch.dict(
        os.environ,
        {
            "LANGFUSE_PUBLIC_KEY": "public",
            "LANGFUSE_SECRET_KEY": "secret",
            "LANGFUSE_TRACING_ENABLED": "true",
        },
        clear=True,
    ):
        configured = _configure_langfuse()

        assert configured is True
        assert os.environ["LANGFUSE_TRACING_ENABLED"] == "true"


def test_no_op_client_and_decorator_preserve_application_behavior():
    client = get_client()
    client.update_current_span(input={"question": "q"})
    client.update_current_generation(output={"content": "a"})
    client.flush()

    def example() -> str:
        return "ok"

    assert observe(example) is example
    assert observe(name="example")(example) is example
    assert example() == "ok"
