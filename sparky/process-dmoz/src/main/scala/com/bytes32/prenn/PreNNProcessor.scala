package com.bytes32.prenn

import java.net.URI

import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.jsoup.Jsoup

import scala.util.matching.Regex

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
      val rawHtml = spark.read.json(rawHtmlWithCategoriesPath.toString)
      extractTextFromRawHtmlWithCategories(rawHtml)
        .write
        .option("compression", "snappy")
        .parquet(out.toString)
    }
  }

  def extractTextFromRawHtmlWithCategories(rawHtml: DataFrame)(implicit spark: SparkSession): Dataset[WebSiteCategoriesText] = {
    import spark.implicits._
    val expectedFields = List("orig_domain", "domain", "categories", "text")
    rawHtml.flatMap {
      r: Row =>
        expectedFields.map(name => r.getAs[String](name)) match {
          case orig_domain :: domain :: categories :: text :: Nil if text.nonEmpty =>
            try {
              val innerHtmlText = Jsoup.parse(text).text()
              List(WebSiteCategoriesText(domain, orig_domain, categories.split("\\|"), innerHtmlText))
            } catch {
              case _: Exception =>
                List.empty[WebSiteCategoriesText]
            }
          case _ => List.empty[WebSiteCategoriesText]
        }
    }
  }

  def filterOutNonEnglishDomains(webSitesWithCategories: Dataset[WebSiteCategoriesText]): Dataset[WebSiteCategoriesText] = {
    webSitesWithCategories
      .filter(isUriEnglishSpeakingDomain(_))
  }

  val EnglishSpeakingHosts: Regex = (
    "\\.com\\.au|\\.net\\.au|" +
      "\\.org\\.au|\\.id\\.au|" +
      "\\.com|\\.gov|\\.co\\.uk|" +
      "\\.uk|\\.gov\\.uk|\\.org\\.uk|" +
      "\\.ie|\\.nz").r.unanchored

  /**
    * This is brutal
    */
  def isUriEnglishSpeakingDomain(w: WebSiteCategoriesText): Boolean = {
    try {
      EnglishSpeakingHosts.findFirstIn(new URI(w.uri).getHost).isDefined
    } catch {
      case _: Exception =>
        false
    }
  }

}
