import logging
import os

# Disable Langfuse tracing during tests so unit-test calls (e.g. generate("hello")
# and deliberate error cases) don't pollute the real Langfuse project.
os.environ["LANGFUSE_TRACING_ENABLED"] = "False"


class _NoActiveLangfuseSpanFilter(logging.Filter):
    """Suppress expected no-span noise while retaining other Langfuse warnings."""

    MESSAGE_PREFIX = "Context error: No active span in current context."

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(self.MESSAGE_PREFIX)


logging.getLogger("langfuse").addFilter(_NoActiveLangfuseSpanFilter())
