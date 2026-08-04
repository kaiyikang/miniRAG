from pathlib import Path

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.readers.base import BaseReader
from llama_index.readers.file import MarkdownReader

from minirag.types import Chunk
from minirag.chunking import clean_markdown, Chunker


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


def local_chunks(doc_dir: str, chunker: Chunker) -> list[Chunk]:
    return _chunk_documents(_load_documents(doc_dir), chunker)
