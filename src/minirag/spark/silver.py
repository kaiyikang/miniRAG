from pyspark.sql import functions as F
from pyspark.sql.window import Window

from minirag.spark.session import get_spark, table_path
from minirag.spark.schemas import (
    VALID_SERVICES,
    VALID_STATUSES,
    VALID_ERROR_CODES,
    VALID_SEVERITIES,
)


def build_silver_tickets(spark):

    bronze = spark.read.format("delta").load(table_path("bronze", "tickets"))

    # Change type
    typed = bronze.withColumn(
        "created_at", F.to_timestamp("created_at", "yyyy-MM-dd'T'HH:mm:ss")
    )

    # not bad is not equal good, null should not be ignored
    reason = (
        F.when(
            F.col("service").isNull() | ~F.col("service").isin(VALID_SERVICES),
            "bad_service",
        )
        .when(
            F.col("error_code").isNull() | ~F.col("error_code").isin(VALID_ERROR_CODES),
            "bad_error_code",
        )
        .when(
            F.col("status").isNull() | ~F.col("status").isin(VALID_STATUSES),
            "bad_status",
        )
        .when(F.col("created_at").isNull(), "bad_timestamp")
        .otherwise(None)
    )

    flagged = typed.withColumn("quarantine_reason", reason)

    valid = flagged.filter(F.col("quarantine_reason").isNull())
    quarantine = flagged.filter(F.col("quarantine_reason").isNotNull())

    # Deduplication: same ticket_id, keep only the most recent entry based on created_at
    w = Window.partitionBy("ticket_id").orderBy(F.col("created_at").desc())
    deduped = (
        valid.withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )

    deduped.write.format("delta").mode("overwrite").save(
        table_path("silver", "tickets")
    )
    quarantine.write.format("delta").mode("overwrite").option(
        "overwriteSchema", True
    ).save(table_path("silver", "quarantine_tickets"))

    return deduped.count(), quarantine.count()


def build_silver_events(spark):

    bronze = spark.read.format("delta").load(table_path("bronze", "events"))
    typed = bronze.withColumn(
        "event_timestamp", F.to_timestamp("event_timestamp", "yyyy-MM-dd'T'HH:mm:ss")
    )

    reason = (
        F.when(
            F.col("severity").isNull() | ~F.col("severity").isin(VALID_SEVERITIES),
            "bad_severity",
        )
        .when(F.col("event_timestamp").isNull(), "bad_timestamp")
        .otherwise(None)
    )

    flagged = typed.withColumn("quarantine_reason", reason)
    valid = flagged.filter(F.col("quarantine_reason").isNull())
    field_bad = flagged.filter(F.col("quarantine_reason").isNotNull())

    # Ticket id in events must also exist in silver_tickets
    ticket_ids = (
        spark.read.format("delta")
        .load(table_path("silver", "tickets"))
        .select("ticket_id")
        .distinct()
    )
    orphans = valid.join(ticket_ids, "ticket_id", "left_anti").withColumn(
        "quarantine_reason", F.lit("orphan_ticket")
    )
    clean = valid.join(ticket_ids, "ticket_id", "left_semi")
    # Good
    clean.write.format("delta").mode("overwrite").save(table_path("silver", "events"))
    # Bad: Stack two DataFrames vertically
    field_bad.unionByName(orphans).write.format("delta").mode("overwrite").option(
        "overwriteSchema", True
    ).save(table_path("silver", "quarantine_events"))

    return clean.count(), field_bad.count() + orphans.count()


if __name__ == "__main__":
    spark = get_spark("silver")
    tv, tq = build_silver_tickets(spark)  # should be done before events
    ev, eq = build_silver_events(spark)
    print(
        f"OK silver tickets(valid={tv} quarantine={tq}) events(clean={ev} quarantine={eq})"
    )
    spark.stop()
