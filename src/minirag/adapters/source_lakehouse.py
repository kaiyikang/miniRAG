from pathlib import Path

from deltalake import DeltaTable

from minirag.domain.models import Chunk
from minirag.domain.ports import DocumentSource

# no JVM without importing pyspark
LAKEHOUSE_ROOT = Path("data/lakehouse")


def _table_path(layer: str, name: str) -> str:
    return str(LAKEHOUSE_ROOT / layer / name)


def read_delta(layer: str, name: str):
    return DeltaTable(_table_path(layer, name)).to_pandas()


class LakehouseSource(DocumentSource):
    """Chunks read from a Delta lakehouse table (delta-rs, no JVM)."""

    def __init__(self, layer: str = "gold", name: str = "rag_chunks"):
        self._layer = layer
        self._name = name

    def load(self) -> list[Chunk]:
        df = read_delta(self._layer, self._name)
        return [
            Chunk(
                document=row.chunk_text,
                metadata={
                    "chunk_id": row.chunk_id,
                    "source_uri": row.source_uri,
                    "document_id": row.document_id,
                    "chunk_index": int(row.chunk_index),
                },
                embedding=None,
            )
            for row in df.itertuples()
        ]
