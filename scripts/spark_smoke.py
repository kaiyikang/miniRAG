"""smoke test: prove Spark + Delta write/read works locally.

Run: uv run --group spark python scripts/spark_smoke.py
"""

from minirag.spark.session import get_spark, table_path


def main() -> None:
    spark = get_spark("smoke")

    df = spark.createDataFrame(
        [("T-101", "open"), ("T-102", "in_progress")],
        ["ticket_id", "status"],
    )

    path = table_path("bronze", "_smoke")
    df.write.format("delta").mode("overwrite").save(path)

    back = spark.read.format("delta").load(path)
    rows = back.count()
    back.show()

    assert rows == 2, f"expected 2 rows, got {rows}"
    print("OK-DELTA-SMOKE")
    spark.stop()


if __name__ == "__main__":
    main()
