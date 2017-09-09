package com.bytes32.prenn

import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.{Dataset, Row, SparkSession}

import scala.util.matching.Regex

/**
  * Created by murariuf on 30/04/2017.
  * Load and clean the raw dataset and return only the text that looks english
  * map the categories against the text
  */
object PreNNProcessor extends HasSpark with JobRunner with LazyLogging with ProcessorConfig with TextCleanup{

  val Word: Regex = "[a-zA-Z]+".r

  case class FilterCategory(name: String, includes: Set[String], excludes: Set[String])

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


  def main(args: Array[String]): Unit = {

    val Config(websitesRawInput, websitesTextOutput, categoriesPath, websitesCleanOutput, local) = parseArgs(args)

    implicit val spark = makeSparkSession("PreNNProcessor", local)
    import spark.implicits._
    /* filter non-english websites */
    runForOutput(websitesTextOutput) {
      val rawHtml = spark.read.json(websitesRawInput)
      (extractTextFromRawHtmlWithCategories _ andThen excludeNonEnglishWebsites)(rawHtml)
        .write
        .option("compression", "snappy")
        .parquet(websitesTextOutput)
    }

    /* map to our categories */
    runForOutput(websitesCleanOutput) {
      val categories = loadCategories(categoriesPath)
      val dmozTextCats = spark.read
        .parquet(websitesTextOutput)
        .as[WebSiteCategoriesText]

      filterAndExpandWebSites(categories)(dmozTextCats)
        .write
        .option("compression", "snappy")
        .parquet(websitesCleanOutput)
    }
  }

}
