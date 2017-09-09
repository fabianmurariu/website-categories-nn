package com.bytes32.v2

import com.bytes32.prenn._
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.{Dataset, SparkSession}

object PreNNProcessor extends HasSpark with JobRunner with LazyLogging with ProcessorConfig with TextCleanup {

  def truncateOrigCategories(cats: Dataset[WebSiteCategoriesText])(implicit spark: SparkSession): Dataset[WebSiteCategoriesTextV2] = {
    import spark.implicits._
    cats.flatMap {
      case WebSiteCategoriesText(uri, origUri, categories, text, _) =>
        categories.flatMap {
          cat =>
            val allCats = cat.stripPrefix("Top/").split("/").toList
            allCats match {
              case one :: two :: three :: tail_* => List(DMOZCats(one, Some(two), Some(three)))
              case one :: two :: tail_* => List(DMOZCats(one, Some(two), None))
              case one :: tail_* => List(DMOZCats(one, None, None))
              case _ => Nil
            }
        }.map { dmozCat => WebSiteCategoriesTextV2(uri, origUri, text, dmozCat) }
    }
  }

  def main(args: Array[String]): Unit = {
    val Config(websitesRawInput, websitesTextOutput, _, websitesCleanOutput, local) = parseArgs(args)

    implicit val spark = makeSparkSession("PreNNProcessor", local)

    /* filter non-english websites */
    runForOutput(websitesTextOutput) {
      val rawHtml = spark.read.json(websitesRawInput)
      (extractTextFromRawHtmlWithCategories _ andThen excludeNonEnglishWebsites) (rawHtml)
        .write
        .option("compression", "snappy")
        .parquet(websitesTextOutput)
    }

  }

}
