# Load .env into os.environ so SDKs that read env directly (e.g. Langfuse) see keys.
from dotenv import load_dotenv

load_dotenv()
