from pyspark.sql import functions as F
from minirag.spark.session import get_spark, table_path

spark = get_spark("explain")

# close broadcast manually to see sortMergeJoin
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

tickets = (
    spark.read.format("delta")
    .load(table_path("silver", "tickets"))
    .select("ticket_id", "service")
)
events = spark.read.format("delta").load(table_path("silver", "events"))

print("=" * 30, "DEFAULT JOIN (expect SortMergeJoin + Exchange)")
events.join(tickets, "ticket_id").explain("formatted")


print("=" * 30, "BROADCAST HINT (expect BroadcastHashJoin, no shuffle on small side)")
events.join(F.broadcast(tickets), "ticket_id").explain("formatted")


spark.stop()
