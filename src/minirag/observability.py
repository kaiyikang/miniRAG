import os
from collections.abc import Callable
from typing import Any


def _configure_langfuse() -> bool:
    """Use Langfuse's no-op mode when credentials are not configured."""
    credentials_configured = bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )
    if not credentials_configured:
        os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
    return credentials_configured


_LANGFUSE_CONFIGURED = _configure_langfuse()

if _LANGFUSE_CONFIGURED:
    # Import only after configuring the environment. Langfuse reads these
    # values when it creates its singleton client.
    from langfuse import get_client, observe
else:

    class _NoOpLangfuseClient:
        def update_current_span(self, *args: Any, **kwargs: Any) -> None:
            pass

        def update_current_generation(self, *args: Any, **kwargs: Any) -> None:
            pass

        def flush(self) -> None:
            pass

    _NO_OP_CLIENT = _NoOpLangfuseClient()

    def get_client() -> _NoOpLangfuseClient:
        return _NO_OP_CLIENT

    def observe[F: Callable[..., Any]](
        func: F | None = None,
        **kwargs: Any,
    ) -> F | Callable[[F], F]:
        if func is not None:
            return func

        def decorator(inner: F) -> F:
            return inner

        return decorator

__all__ = ["get_client", "observe"]
