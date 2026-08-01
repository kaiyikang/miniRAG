"""Local Spark + Delta session for the offline data plane.

This is the entry point for every pipeline job (bronze/silver/gold). It is
deliberately local-only: the whole point of the exercise is to run the full
lakehouse lifecycle on one machine, no Databricks workspace required.
"""

from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

# Where Delta tables live on local disk. Mirrors a catalog: <layer>/<table>.
LAKEHOUSE_ROOT = Path("data/lakehouse")


def get_spark(app_name: str = "minirag-lakehouse") -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        # Mount Delta: SQL extension (MERGE, time travel) + catalog impl.
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # ponytail: 4 shuffle partitions, not the default 200 — single machine,
        # tiny data. Bump if data ever grows past a few GB.
        .config("spark.sql.shuffle.partitions", "4")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def table_path(layer: str, name: str) -> str:
    """Local path for a Delta table, e.g. table_path('bronze', 'documents')."""
    return str(LAKEHOUSE_ROOT / layer / name)
