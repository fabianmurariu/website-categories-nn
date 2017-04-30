package com.bytes32.prenn

import java.net.URI

import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.jsoup.Jsoup

/**
  * Created by murariuf on 30/04/2017.
  */
object PreNNProcessor extends HasSpark with JobRunner with LazyLogging {

  case class WebSiteCategoriesText(uri: String, origUri: String, categories: Seq[String], text: String)

  def main(args: Array[String]): Unit = {
    val (rawHtmlWithCategoriesPath, out, local) = args.toList match {
      case path :: output :: localFlag :: _ => (new URI(path), new URI(output), localFlag.toBoolean)
      case _ => throw new IllegalArgumentException("Missing or invalid path expected raw categories and html file and output path")
    }

    implicit val spark = makeSparkSession("PreNNProcessor", local)
    runForOutput(out.getPath) {
      val rawHtml = spark.read.json(rawHtmlWithCategoriesPath.getPath)
      extractTextFromRawHtmlWithCategories(rawHtml).show
    }
  }

  def extractTextFromRawHtmlWithCategories(rawHtml: DataFrame)(implicit spark: SparkSession): Dataset[WebSiteCategoriesText] = {
    import spark.implicits._
    val expectedFields = List("orig_domain", "domain", "categories", "text")
    rawHtml.flatMap {
      r: Row =>
        expectedFields.map(name => r.getAs[String](name)) match {
          case orig_domain :: domain :: categories :: text :: Nil =>
            List(WebSiteCategoriesText(domain, orig_domain, categories.split("\\|"), Jsoup.parse(text).text()))
          case _ => List.empty[WebSiteCategoriesText]
        }
    }
  }

}
