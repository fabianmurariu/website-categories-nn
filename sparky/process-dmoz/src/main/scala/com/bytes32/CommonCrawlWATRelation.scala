package com.bytes32

import java.io.DataInputStream

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.sources.{BaseRelation, TableScan}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import fs2.io
class CommonCrawlWATRelation(val location: String,
                             @transient val sqlContext: SQLContext) extends BaseRelation with TableScan with Serializable {

  override def schema: StructType = StructType(Seq(
    StructField("WARC_Version", StringType(), nullable = true)))

  override def buildScan(): RDD[Row] = {
    sqlContext.sparkContext
      .binaryFiles(location)
      .flatMap{
        case (file, stream) =>
          val dataStream: DataInputStream = stream.open()

          List()
      }
  }
}
