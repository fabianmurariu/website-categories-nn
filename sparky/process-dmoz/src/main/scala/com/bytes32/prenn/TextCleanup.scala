package com.bytes32.prenn

import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.jsoup.Jsoup

trait TextCleanup {

  def excludeNonEnglishWebsites(ws: Dataset[WebSiteCategoriesText])
                               (implicit spark: SparkSession): Dataset[WebSiteCategoriesText] = {

    ws.filter { ws: WebSiteCategoriesText => Language.detectEnglish(ws.text).isDefined }
  }

  def extractTextFromRawHtmlWithCategories(rawHtml: DataFrame)(implicit spark: SparkSession): Dataset[WebSiteCategoriesText] = {
    import spark.implicits._
    val expectedFields = List("orig_domain", "domain", "categories", "text")
    rawHtml.flatMap {
      r: Row =>
        expectedFields.map(name => r.getAs[String](name)) match {
          case orig_domain :: domain :: categories :: text :: Nil if text != null && text.nonEmpty =>
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


}
