from pyspark.sql.types import StringType, StructField, StructType

# Bronze: all read as string
TICKETS_RAW_SCHEMA = StructType(
    [
        StructField("ticket_id", StringType()),
        StructField("service", StringType()),
        StructField("error_code", StringType()),
        StructField("priority", StringType()),
        StructField("status", StringType()),
        StructField("created_at", StringType()),
        StructField("assignee", StringType()),
    ]
)

EVENTS_RAW_SCHEMA = StructType(
    [
        StructField("event_id", StringType()),
        StructField("ticket_id", StringType()),
        StructField("event_type", StringType()),
        StructField("note", StringType()),
        StructField("severity", StringType()),
        StructField("event_timestamp", StringType()),
    ]
)

# Silver legal for verification
VALID_SERVICES = ["auth-service", "payment-gateway", "search-index"]
VALID_STATUSES = ["open", "in_progress", "resolved", "closed"]
VALID_ERROR_CODES = ["E104", "E205", "E301", "E402"]
VALID_SEVERITIES = ["critical", "high", "medium", "low"]
