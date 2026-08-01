from pyspark.sql import functions as F

from minirag.spark.session import get_spark, table_path
from minirag.spark.schemas import TICKETS_RAW_SCHEMA, EVENTS_RAW_SCHEMA


def _with_ingest_metadata(df):
    raw_cols = df.columns  # for content hash
    return (
        df.withColumn("source_file", F.input_file_name())
        .withColumn("ingested_at", F.current_timestamp())
        .withColumn("content_hash", F.sha2(F.concat_ws("|", *raw_cols), 256))
    )


def build_bronze(spark):
    tickets = (
        spark.read.schema(TICKETS_RAW_SCHEMA)
        .option("header", True)
        .csv("data/raw/tickets.csv")
    )
    tickets = _with_ingest_metadata(tickets)

    tickets.write.format("delta").mode("overwrite").save(
        table_path("bronze", "tickets")
    )

    events = (
        spark.read.schema(EVENTS_RAW_SCHEMA)
        .option("multiline", True)
        .json("data/raw/ticket_events.json")
    )
    events = _with_ingest_metadata(events)

    events.write.format("delta").mode("overwrite").save(table_path("bronze", "events"))

    return tickets.count(), events.count()


if __name__ == "__main__":
    spark = get_spark("bronze")
    t, e = build_bronze(spark)
    print(f"OK bronze tickets={t} events={e}")
    spark.stop()
