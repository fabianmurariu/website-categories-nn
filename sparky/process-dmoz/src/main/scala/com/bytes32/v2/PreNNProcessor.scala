package com.bytes32.v2

import com.bytes32.prenn._
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.{Dataset, SparkSession}

object PreNNProcessor extends HasSpark with JobRunner with LazyLogging with ProcessorConfig with TextCleanup {

  def truncateOrigCategoriesToTop3(cats: Dataset[WebSiteCategoriesText])(implicit spark: SparkSession): Dataset[WebSiteCategoriesText] = {
    import spark.implicits._
    cats.flatMap {
      case WebSiteCategoriesText(uri, origUri, categories, text, _) =>
        categories.flatMap {
          cat =>
            val allCats = cat.stripPrefix("Top/").split("/").toList
            allCats match {
              case one :: two :: three :: _ => List(DMOZCats(one, Some(two), Some(three)))
              case one :: two :: _ => List(DMOZCats(one, Some(two), None))
              case one :: _ => List(DMOZCats(one, None, None))
              case _ => Nil
            }
        }.map { cs =>
          val cat = (List(cs.top) ++ List(cs.cat2, cs.cat3).map(_.getOrElse("")).filterNot(_ == "")).mkString("/")
          WebSiteCategoriesText(uri, origUri, Seq(cat).filterNot(_ == ""), text)
        }.filterNot(_.categories.isEmpty)
    }
  }

  def breakIntoSentences(sentenceLength: Int)
                        (cats: Dataset[WebSiteCategoriesText])
                        (implicit spark: SparkSession): Dataset[WebSiteCategoriesText] = {
    import spark.implicits._
    cats.flatMap{
      case WebSiteCategoriesText(uri, origUri, categories, text, origCategories) =>
        Text.splitAndClean(text)
          .sliding(sentenceLength, sentenceLength).map{
          tks => WebSiteCategoriesText(uri, origUri, categories, tks.mkString(" "), origCategories)
        }
    }
  }

  def main(args: Array[String]): Unit = {
    val Config(websitesRawInput, websitesTextOutput, _, websitesCleanOutput, local) = parseArgs(args)

    implicit val spark: SparkSession = makeSparkSession("PreNNProcessor", local)
    import spark.implicits._

    /* filter non-english websites */
    runForOutput(websitesTextOutput) {
      val rawHtml = spark.read.json(websitesRawInput)
      (extractTextFromRawHtmlWithCategories _ andThen excludeNonEnglishWebsites) (rawHtml)
        .write
        .option("compression", "snappy")
        .parquet(websitesTextOutput)
    }

    runForOutput(websitesCleanOutput) {
      val dmozTextCats = spark.read
        .parquet(websitesTextOutput)
        .as[WebSiteCategoriesText]

      (truncateOrigCategoriesToTop3 _ andThen breakIntoSentences(128)) (dmozTextCats)
        .write
        .option("compression", "snappy")
        .parquet(websitesCleanOutput)
    }
  }

}
