from minirag.adapters.source_lakehouse import read_delta


class TicketContextTool:
    """Give the gold_ticket_context service to the agent. Read with delta-rs, no JVM."""

    def __init__(self):
        self._df = read_delta("gold", "ticket_context")

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
