import re
from abc import ABC, abstractmethod

_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def clean_markdown(text: str) -> str:
    text = _FRONTMATTER_RE.sub("", text or "")
    text = _IMAGE_RE.sub("", text)
    return _LINK_RE.sub(r"\1", text)


def chunk_by_words(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    if size <= overlap:
        raise ValueError("size must be greater than overlap")
    words = (text or "").split()
    if not words:
        return []
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, len(words), step)]


class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """text -> list of chunk strings"""


class SlidingWindowChunker(Chunker):
    def __init__(self, chunk_size: int = 200, overlap: int = 40):
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        return chunk_by_words(text, self.chunk_size, self.overlap)


class ParagraphChunker(Chunker):
    def chunk(self, text: str) -> list[str]:
        return [p.strip() for p in (text or "").split("\n\n") if p.strip()]


class SpacyChunker(Chunker):
    """python -m spacy download en_core_web_md"""

    def __init__(self, model_name: str = "en_core_web_md"):
        import spacy

        self.model = spacy.load(model_name, disable=["ner"])

    def chunk(self, text: str) -> list[str]:
        return [s.text for s in self.model(text).sents]
