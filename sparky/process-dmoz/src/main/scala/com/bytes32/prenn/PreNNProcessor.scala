package com.bytes32.prenn

import java.net.URI
import java.util.zip.GZIPInputStream

import com.optimaize.langdetect.LanguageDetectorBuilder
import com.optimaize.langdetect.i18n.LdLocale
import com.optimaize.langdetect.ngram.NgramExtractors
import com.optimaize.langdetect.profiles.LanguageProfileReader
import com.optimaize.langdetect.text.CommonTextObjectFactories
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.jsoup.Jsoup

import scala.collection.JavaConversions._
import scala.io.Source
import scala.util.matching.Regex

/**
  * Created by murariuf on 30/04/2017.
  * Load and clean the raw dataset and return only the text that looks english
  * map the categories against the text
  */
object PreNNProcessor extends HasSpark with JobRunner with LazyLogging {

  val Word: Regex = "[a-zA-Z]+".r

  case class Config(websitesRawInput: String, websitesTextOutput: String, categoriesPath: String, websitesCleanOutput: String, local: Boolean = false)

  case class WebSiteCategoriesText(uri: String, origUri: String, categories: Seq[String], text: String, origCategories:Seq[String] = Seq.empty)

  case class FilterCategory(name: String, includes: Set[String], excludes: Set[String])

  def excludeNonEnglishWebsites(ws: Dataset[WebSiteCategoriesText])
                               (implicit spark: SparkSession): Dataset[WebSiteCategoriesText] = {

    ws.filter { ws: WebSiteCategoriesText => Language.detectEnglish(ws.text).isDefined }
  }

  def checkCategory(dmozCat: String)(filterCat: String): Boolean = {
    dmozCat.stripPrefix("Top/").startsWith(filterCat)
  }

  def filterAndExpandWebSites(filterCategories: Seq[FilterCategory])(data: Dataset[WebSiteCategoriesText])
                             (implicit spark: SparkSession): Dataset[WebSiteCategoriesText] = {
    import spark.implicits._

    data.flatMap {
      case WebSiteCategoriesText(uri, origUri, cats: Seq[String], text, _) =>
        for {
          dmozCat: String <- cats
          categories: FilterCategory <- filterCategories
          if categories.includes.exists(checkCategory(dmozCat)) && !categories.excludes.exists(checkCategory(dmozCat))
        } yield WebSiteCategoriesText(uri, origUri, List(categories.name), text, cats)
    }
  }

  def loadCategories(catsPath: String)(implicit spark: SparkSession): Seq[FilterCategory] = {
    spark.read.json(catsPath).rdd.map {
      case Row(filters: Row, name: String) =>
        val includes = Option(filters.getAs[Seq[String]]("includes")).getOrElse(Seq.empty[String])
        val excludes = Option(filters.getAs[Seq[String]]("excludes")).getOrElse(Seq.empty[String])
        FilterCategory(name, includes.toSet, excludes.toSet)
    }.collect()
  }

  def parseArgs(args: Array[String]): Config = {
    val parser = new scopt.OptionParser[Config](getClass.getSimpleName) {
      opt[Unit]("local").action((_, config) =>
        config.copy(local = true)
      )
      opt[String]("websitesRawInput").required().action((path, config) =>
        config.copy(websitesRawInput = path)).text("Path to the dmoz corpus with static webpages crawled")
      opt[String]("websitesTextOutput").required().action((path, config) =>
        config.copy(websitesTextOutput = path)).text("Path to output clean text for every webpage")
      opt[String]("categoriesPath").optional().action((path, config) =>
        config.copy(categoriesPath = path)).text("Path to categories mapping as json lines")
      opt[String]("websitesCleanOutput").optional().action((path, config) =>
        config.copy(websitesCleanOutput = path)).text("Path to output of categories and tokens as array")

      override def reportError(msg: String): Unit = throw new IllegalArgumentException(s"$msg\n$usage")
    }

    parser.parse(args, Config(null, null, null, null)).get
  }

  def main(args: Array[String]): Unit = {

    val Config(websitesRawInput, websitesTextOutput, categoriesPath, websitesCleanOutput, local) = parseArgs(args)

    implicit val spark = makeSparkSession("PreNNProcessor", local)
    import spark.implicits._
    /* extract the words from the websites */
    runForOutput(websitesTextOutput) {
      val rawHtml = spark.read.json(websitesRawInput)
      extractTextFromRawHtmlWithCategories(rawHtml)
        .write
        .option("compression", "snappy")
        .parquet(websitesTextOutput)
    }

    /* convert to local categories and cleanup non-english words */
    runForOutput(websitesCleanOutput) {
      val categories = loadCategories(categoriesPath)
      val dmozTextCats = spark.read
        .parquet(websitesTextOutput)
        .as[WebSiteCategoriesText]

      val wordTokens: Dataset[WebSiteCategoriesText] = (filterAndExpandWebSites(categories) _ andThen
        excludeNonEnglishWebsites).apply(dmozTextCats)

      wordTokens.write.option("compression", "snappy").parquet(websitesCleanOutput)
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
