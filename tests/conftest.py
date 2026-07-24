import os

# Disable Langfuse tracing during tests so unit-test calls (e.g. generate("hello")
# and deliberate error cases) don't pollute the real Langfuse project.
os.environ["LANGFUSE_TRACING_ENABLED"] = "False"
