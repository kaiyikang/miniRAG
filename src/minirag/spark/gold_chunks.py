from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

from delta.tables import DeltaTable
from minirag.spark.session import get_spark, table_path


# sliding window split the chunk with the words
# in production, the logic can be shared
def _chunk_text(text, size=200, overlap=40):
    words = (text or "").split()
    if not words:
        return []
    chunks, start, step = [], 0, size - overlap
    while start < len(words):
        chunks.append(" ".join(words[start : start + size]))
        start += step
    return chunks


def _prepare_chunks(spark):
    # wholetext = True, one file one line
    docs = (
        spark.read.text("data/raw/docs", wholetext=True)
        .withColumn("source_uri", F.input_file_name())
        .withColumnRenamed("value", "clean_text")
        .withColumn("document_id", F.sha2(F.col("source_uri"), 256))  # stable
        .withColumn("content_hash", F.sha2(F.col("clean_text"), 256))  # can be updated
    )

    chunk_udf = F.udf(_chunk_text, ArrayType(StringType()))

    exploded = docs.withColumn("chunks", chunk_udf("clean_text")).select(
        "document_id",
        "content_hash",
        "source_uri",
        F.posexplode(F.col("chunks")).alias("chunk_index", "chunk_text"),
    )

    return exploded.withColumn(
        "chunk_id",
        F.sha2(
            F.concat_ws(
                "|", "document_id", "content_hash", F.col("chunk_index").cast("string")
            ),
            256,
        ),
    )


# DeltaTable.forPath(spark, path).history(1) cannot be used,
# since a MERGE with "nothing inserted" will not generate a new commit in Delta.
def build_gold_rag_chunks(spark):
    path = table_path("gold", "rag_chunks")
    chunks = _prepare_chunks(spark)

    if DeltaTable.isDeltaTable(spark, path):
        before = spark.read.format("delta").load(path).count()
        dt = DeltaTable.forPath(spark, path)
        dt.alias("t").merge(
            chunks.alias("s"), "t.chunk_id = s.chunk_id"
        ).whenNotMatchedInsertAll().execute()
        total = spark.read.format("delta").load(path).count()
        inserted = total - before
    else:
        chunks.write.format("delta").save(path)
        total = spark.read.format("delta").load(path).count()
        inserted = total

    return total, inserted


if __name__ == "__main__":
    spark = get_spark("gold_chunks")
    total, inserted = build_gold_rag_chunks(spark)
    print(f"OK gold rag_chunks total={total} inserted_this_run={inserted}")
    spark.stop()
