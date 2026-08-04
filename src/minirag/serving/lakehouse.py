from pathlib import Path

from deltalake import DeltaTable

from minirag.domain.models import Chunk

# no JVM without importing pyspark
LAKEHOUSE_ROOT = Path("data/lakehouse")


def _table_path(layer: str, name: str) -> str:
    return str(LAKEHOUSE_ROOT / layer / name)


def _read_delta(layer: str, name: str):
    return DeltaTable(_table_path(layer, name)).to_pandas()


# DTO converter
def lakehouse_chunks(layer: str = "gold", name: str = "rag_chunks") -> list[Chunk]:
    df = _read_delta(layer, name)

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


class TicketContextTool:
    """Give the gold_ticket_context service to the agent. Read with delta-rs, no JVM."""

    def __init__(self):
        self._df = _read_delta("gold", "ticket_context")

    def get_context(self, ticket_id: str) -> dict | None:
        rows = self._df[self._df["ticket_id"] == ticket_id]
        if rows.empty:
            return None

        r = rows.iloc[0]
        return {
            "ticket_id": r["ticket_id"],
            "service": r["service"],
            "error_code": r["error_code"],
            "status": r["status"],
            "critical_event_count": int(r["critical_event_count"]),
            "latest_event_severity": r["latest_event_severity"],
            "latest_event_note": r["latest_event_note"],
        }


if __name__ == "__main__":
    t = TicketContextTool()
    print("T-102 ->", t.get_context("T-102"))
    assert t.get_context("T-102")["critical_event_count"] == 2
    assert t.get_context("T-000") is None
    print("OK ticket_context_tool")
