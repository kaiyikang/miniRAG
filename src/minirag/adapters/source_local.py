from pathlib import Path

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.readers.base import BaseReader
from llama_index.readers.file import MarkdownReader

from minirag.domain.models import Chunk
from minirag.adapters.chunker import clean_markdown
from minirag.domain.ports import Chunker, DocumentSource


class LocalMarkdownSource(DocumentSource):
    def __init__(self, doc_dir: str, chunker: Chunker):
        self._doc_dir = doc_dir
        self._chunker = chunker

    def load(self) -> list[Chunk]:
        return _chunk_documents(_load_documents(self._doc_dir), self._chunker)


class MarkdownWithoutFrontmatterReader(BaseReader):
    def load_data(self, file_path, extra_info=None):
        docs = MarkdownReader(remove_images=False, remove_hyperlinks=False).load_data(
            file_path, extra_info=extra_info
        )
        return [
            Document(text=clean_markdown(d.text), metadata=d.metadata) for d in docs
        ]


def _load_documents(path: str) -> list[Document]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Document path not found: {path}")
    reader = SimpleDirectoryReader(
        input_dir=path,
        recursive=True,
        file_extractor={".md": MarkdownWithoutFrontmatterReader()},
    )
    return reader.load_data()


def _chunk_documents(docs: list[Document], chunker: Chunker) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        md = doc.metadata
        for idx, text in enumerate(chunker.chunk(doc.text)):
            chunks.append(
                Chunk(
                    document=text,
                    metadata={
                        "chunk_idx": idx,
                        "file_name": md.get("file_name", "unknown"),
                        "file_path": md.get("file_path", "unknown"),
                    },
                    embedding=None,
                )
            )
    return chunks
