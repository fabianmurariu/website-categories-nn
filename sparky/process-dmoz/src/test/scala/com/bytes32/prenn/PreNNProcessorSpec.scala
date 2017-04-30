package com.bytes32.prenn

import org.apache.spark.sql.SparkSession
import org.scalatest.{FlatSpec, Matchers}

/**
  * Created by murariuf on 30/04/2017.
  */
class PreNNProcessorSpec extends FlatSpec with Matchers with HasSpark {

  import PreNNProcessor._

  implicit val spark: SparkSession = makeSparkSession("test", local = true)

  "PreNNProcessor" should "extract content from raw html and expand categories in a seq" in {
    import spark.implicits._
    val sample = Seq(
      ("www.blerg.com", "http://www.blerg.com", "some|other","""<html><head><title>hello</title></head><body><div>some text here</div></body></html>""")
    ).toDF("orig_domain", "domain", "categories", "text")
    val actual = extractTextFromRawHtmlWithCategories(sample).collect()
    actual should contain theSameElementsAs List(WebSiteCategoriesText("http://www.blerg.com", "www.blerg.com", Seq("some", "other"), "hello some text here"))
  }

  it should "filter out non english domains" in {
    import spark.implicits._
    val samples = Seq(
      WebSiteCategoriesText("http://www.blerg.com", "www.blerg.com", Seq("some", "other"), "hello some text here"),
      WebSiteCategoriesText("http://www.blerg.co.uk", "www.blerg.com", Seq("some", "other"), "hello some text here"),
      WebSiteCategoriesText("http://www.blerg.fr", "www.blerg.fr", Seq("some", "other"), "les francais")).toDS()

    val actual = filterOutNonEnglishDomains(samples).collect()
    actual should contain theSameElementsAs List(
      WebSiteCategoriesText("http://www.blerg.co.uk", "www.blerg.com", Seq("some", "other"), "hello some text here"),
      WebSiteCategoriesText("http://www.blerg.com", "www.blerg.com", Seq("some", "other"), "hello some text here"))
  }

}
