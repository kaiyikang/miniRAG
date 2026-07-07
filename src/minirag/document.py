from minirag.types import Chunk
from llama_index.core import SimpleDirectoryReader, Document
from abc import ABC, abstractmethod
from llama_index.core.readers.base import BaseReader
from llama_index.readers.file import MarkdownReader
import re
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


class ParagraphChunker(Chunker):
    def chunk(self, text: str) -> list[str]:
        return [p.strip() for p in text.split("\n\n") if p.strip()]


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

        chunks = []
        i = 0
        n = len(text)
        while i < n:
            end = min(i + self.chunk_size, n)
            if end < n:
                snap = text.rfind(" ", i, end)
                if snap > i:
                    end = snap
            chunks.append(text[i:end])
            if end >= n:
                break
            i = max(end - self.overlap, i + 1)
        return chunks


class MarkdownWithoutFrontmatterReader(BaseReader):
    _frontmatter_re = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)

    def load_data(self, file_path, extra_info=None):
        docs = MarkdownReader().load_data(file_path, extra_info=extra_info)
        return [
            Document(
                text=self._frontmatter_re.sub("", doc.text),
                metadata=doc.metadata,
            )
            for doc in docs
        ]


def load_documents(path: str) -> list[Document]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Document path not found: {path}")
    reader = SimpleDirectoryReader(input_dir=path, recursive=True)
    return reader.load_data()


def chunk_documents(docs: list[Document], chunker: Chunker) -> list[Chunk]:
    return chunker.to_chunks(docs)
