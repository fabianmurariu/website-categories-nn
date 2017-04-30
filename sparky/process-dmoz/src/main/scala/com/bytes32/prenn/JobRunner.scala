package com.bytes32.prenn

import java.net.URI

import com.typesafe.scalalogging.LazyLogging
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

/**
  * Created by murariuf on 30/04/2017.
  */
trait JobRunner { self: LazyLogging =>
  def runForOutput(outputPaths: String*)(run: => Unit)(implicit spark: SparkSession): Unit = {

    def pathIsPopulated(path: String): Boolean = {
      val fs = FileSystem.get(new URI(path), spark.sparkContext.hadoopConfiguration)
      val base = new Path(path)
      fs.exists(base) && (fs.isFile(base) || fs.exists(new Path(path + "/_SUCCESS")))
    }

    def clearOutputs(paths: Seq[String]): Unit =
      paths.foreach { path =>
        val fs = FileSystem.get(new URI(path), spark.sparkContext.hadoopConfiguration)
        if (fs.exists(new Path(path))) {
          logger.info(s"Clearing output path $path")
          fs.delete(new Path(path), true)
        }
      }

    if (outputPaths.forall(pathIsPopulated)) {
      logger.info(s"All outputs populated already, skipping step (paths=$outputPaths)")
    } else {
      clearOutputs(outputPaths)
      run
    }
  }
}
