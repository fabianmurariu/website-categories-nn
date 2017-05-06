package com.bytes32.prenn

import java.net.URI
import java.util.zip.GZIPInputStream

import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.jsoup.Jsoup

import scala.io.Source
import scala.util.matching.Regex

/**
  * Created by murariuf on 30/04/2017.
  */
object PreNNProcessor extends HasSpark with JobRunner with LazyLogging {

  val Word: Regex = "[a-zA-Z]+".r
  case class Config(websitesRawInput:String, websitesTextOutput:String, categoriesPath:String, local:Boolean)
  case class WebSiteCategoriesText(uri: String, origUri: String, categories: Seq[String], text: String)
  case class WebSiteCategoriesTokens(uri: String, category: String, words: Seq[String])
  case class FilterCategory(name: String, includes: Set[String], excludes: Set[String])

  def excludeNonEnglishWebsitesFlatten(ws: Dataset[WebSiteCategoriesText], words: Set[String])
                                      (implicit spark: SparkSession): Dataset[WebSiteCategoriesTokens] = {
    import spark.implicits._
    ws.flatMap {
      case WebSiteCategoriesText(uri, _, categories, text) =>
        val englishOnlyWords = text.split("\\s+").filter(word => words.contains(word))
        if (englishOnlyWords.length > 5) {
          categories.map(cat => WebSiteCategoriesTokens(uri, cat, englishOnlyWords))
        } else List.empty
    }
  }

  def loadEnglishWords: Set[String] = {
    Source
      .fromInputStream(new GZIPInputStream(getClass.getResourceAsStream("/en_words.csv.gz")))
      .getLines()
      .filter(word => Word.findFirstIn(word).isDefined)
      .toSet
  }

  def checkCategory(dmozCat: String)(filterCat: String): Boolean = {
    dmozCat.stripPrefix("Top/").startsWith(filterCat)
  }

  def filterAndExpandWebSites(data: Dataset[WebSiteCategoriesText], filterCategories: Seq[FilterCategory])
                             (implicit spark: SparkSession): Dataset[WebSiteCategoriesText] = {
    import spark.implicits._

    data.flatMap {
      case WebSiteCategoriesText(uri, origUri, categories, text) =>
        for {
          dmozCat: String <- categories
          categories: FilterCategory <- filterCategories
          if categories.includes.exists(checkCategory(dmozCat)) && !categories.excludes.exists(checkCategory(dmozCat))
        } yield WebSiteCategoriesText(uri, origUri, List(categories.name), text)
    }
  }


  def selectFilterCategory(webSitesCategoriesText: WebSiteCategoriesText, filterCategories: Seq[FilterCategory])(category: String): Seq[WebSiteCategoriesText] = {
    filterCategories
      .filter(categoryIsIncluded(category))
      .map(cat => WebSiteCategoriesText(webSitesCategoriesText.uri, webSitesCategoriesText.origUri, List(cat.name), webSitesCategoriesText.text))
  }

  def categoryIsIncluded(category: String)(fCat: FilterCategory): Boolean =
    fCat.includes.exists(cat => category.contains(cat))


  def loadCategories(catsPath: String)(implicit spark: SparkSession): Seq[FilterCategory] = {
    spark.read.json(catsPath).rdd.map {
      case Row(filters: Row, name: String) =>
        val includes = Option(filters.getAs[Seq[String]]("includes")).getOrElse(Seq.empty[String])
        val excludes = Option(filters.getAs[Seq[String]]("excludes")).getOrElse(Seq.empty[String])
        FilterCategory(name, includes.toSet, excludes.toSet)
    }.collect()
  }

  def main(args: Array[String]): Unit = {
    val (rawHtmlWithCategoriesPath, out, out2, local) = args.toList match {
      case path :: output :: output2 :: localFlag :: _ => (new URI(path), new URI(output), new URI(output2), localFlag.toBoolean)
      case _ => throw new IllegalArgumentException("Missing or invalid path expected raw categories and html file and output path")
    }

    implicit val spark = makeSparkSession("PreNNProcessor", local)
    /* extract the words from the websites */
    runForOutput(out.toString) {
      val rawHtml = spark.read.json(rawHtmlWithCategoriesPath.toString)
      extractTextFromRawHtmlWithCategories(rawHtml)
        .write
        .option("compression", "snappy")
        .parquet(out.toString)
    }

    runForOutput(out2.getPath) {
      val categories = loadCategories("/Users/murariuf/Source/website-categories-nn/sparky/process-dmoz/src/main/resources/subcategories.jl")
      val words = loadEnglishWords
      val dmozTextCats = spark.read
        .parquet(out.toString)
        .as[WebSiteCategoriesText]

      val wordTokens = ((filterAndExpandWebSites(_, categories)) andThen
        (excludeNonEnglishWebsitesFlatten(_, words))).apply(dmozTextCats)

      wordTokens.write.option("compression", "snappy").parquet(out2.toString)
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
