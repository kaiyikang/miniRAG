from pyspark.sql import functions as F
from pyspark.sql.window import Window

from minirag.spark.session import get_spark, table_path


def build_gold_ticket_context(spark):
    tickets = spark.read.format("delta").load(table_path("silver", "tickets"))
    events = spark.read.format("delta").load(table_path("silver", "events"))

    w = Window.partitionBy("ticket_id").orderBy(F.col("event_timestamp").desc())
    latest = (
        events.withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .select(
            "ticket_id",
            F.col("note").alias("latest_event_note"),
            F.col("event_timestamp").alias("latest_event_at"),
            F.col("severity").alias(
                "latest_event_severity"
            ),  # Avoid collision after join
        )
    )

    crit = (
        events.filter(F.col("severity") == "critical")
        .groupBy("ticket_id")
        .agg(F.count("*").alias("critical_event_count"))
    )

    gold = (
        tickets.select(
            "ticket_id",
            "service",
            "error_code",
            "priority",
            "status",
            "assignee",
            "created_at",
        )
        .join(latest, "ticket_id", "left")
        .join(crit, "ticket_id", "left")
        .withColumn(
            "critical_event_count", F.coalesce("critical_event_count", F.lit(0))
        )
    )

    gold.write.format("delta").mode("overwrite").save(
        table_path("gold", "ticket_context")
    )
    return gold.count()


if __name__ == "__main__":
    spark = get_spark("gold")
    n = build_gold_ticket_context(spark)
    print(f"OK gold ticket_context rows={n}")
    spark.stop()
