from minirag.types import Chunk
from llama_index.core import SimpleDirectoryReader, Document
from abc import ABC, abstractmethod

from pathlib import Path


class Chunker(ABC):

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """do chunk"""

    def to_chunks(self, docs: list[Document]) -> list[Chunk]:
        chunks = []
        for doc in docs:
            metadata = doc.metadata
            file_name = metadata.get("file_name", "unknown_name")
            file_path = metadata.get("file_path", "unknown_path")
            texts = self.chunk(doc.text)
            for idx, text in enumerate(texts):
                chunks.append(
                    Chunk(
                        document=text,
                        metadata={
                            "chunk_idx": idx,
                            "file_name": file_name,
                            "file_path": file_path,
                        },
                        embedding=None,
                    )
                )
        return chunks


class SpacyChunker(Chunker):
    """python -m spacy download en_core_web_md"""

    def __init__(self, model_name: str = "en_core_web_md"):
        self.model = self._load_model(model_name)

    def _load_model(self, model_name: str):
        try:
            import spacy
        except ImportError:
            raise ImportError("Spacy is not installed!")

        model = spacy.load(model_name, disable=["ner"])
        return model

    def chunk(self, text: str) -> list[str]:
        return [s.text for s in self.model(text).sents]


class SlidingWindowChunker(Chunker):

    def __init__(self, chunk_size: int = 256, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")

        self.step = chunk_size - overlap

    def chunk(self, text: str) -> list[str]:

        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.step)]


def load_documents(path: str) -> list[Document]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Document path not found: {path}")
    reader = SimpleDirectoryReader(input_dir=path)
    return reader.load_data()


def chunk_documents(docs: list[Document], chunker: Chunker) -> list[Chunk]:
    return chunker.to_chunks(docs)
