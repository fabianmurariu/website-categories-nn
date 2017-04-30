package com.bytes32.prenn

import org.apache.spark.sql.SparkSession

/**
  * Created by murariuf on 30/04/2017.
  */
trait HasSpark {

  def makeSparkSession(appName: String, local: Boolean = false): SparkSession = {
    val builder = SparkSession.builder().appName(appName)
    if (local)
      builder.master("local[*]").getOrCreate()
    else builder.getOrCreate()
  }

}
