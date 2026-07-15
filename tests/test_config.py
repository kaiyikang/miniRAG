import os
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic_settings import SettingsConfigDict
from minirag.config import Settings, get_settings


class TestSettings(unittest.TestCase):
    def test_settings_loads_defaults(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                Settings,
                "model_config",
                SettingsConfigDict(env_file=None, extra="ignore"),
            ),
        ):
            s = Settings()
            self.assertIsNone(s.openrouter_api_key)
            self.assertEqual(s.openrouter_model, "z-ai/glm-5.2")
            self.assertEqual(s.openrouter_embed_model, "google/gemini-embedding-2")
            self.assertEqual(s.embedding_model, "all-MiniLM-L6-v2")
            self.assertEqual(
                s.collection_name, "personal_notes_gemini_embedding_2_v1"
            )
            project_root = Path(__file__).resolve().parent.parent
            self.assertEqual(s.documents_dir, str(project_root / "data" / "raw"))
            self.assertEqual(
                s.vector_store_path, str(project_root / "data" / "chroma")
            )

    def test_settings_reads_env_vars(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "secret", "OPENROUTER_MODEL": "custom-model"}):
            s = Settings()
            self.assertEqual(s.openrouter_api_key, "secret")
            self.assertEqual(s.openrouter_model, "custom-model")

    def test_settings_str_returns_json(self):
        s = Settings()
        text = str(s)
        self.assertIn("openrouter_model", text)


class TestGetSettings(unittest.TestCase):
    def test_returns_fresh_instance(self):
        s1 = get_settings()
        s2 = get_settings()
        self.assertIsNot(s1, s2)


if __name__ == "__main__":
    unittest.main()
