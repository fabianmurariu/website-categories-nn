package com.bytes32.v2

import com.bytes32.prenn.{DMOZCats, HasSpark, Text, WebSiteCategoriesText}
import org.apache.spark.sql.SparkSession
import org.scalatest.{FlatSpec, Matchers}

class PreNNProcessorSpec extends FlatSpec with Matchers with HasSpark {

  implicit val spark: SparkSession = makeSparkSession("test", local = true)

  import PreNNProcessor._

  "PreNNProcessor v2" should "truncate the original categories into the top 3 levels" in {
    import spark.implicits._
    val sample = Seq(
      Seq("Top/Blerg"),
      Seq("Top/"),
      Seq("Top/Blergo/Blargo"),
      Seq("Top/Blergo/Blargo/The/Third"),
      Seq("Top/Regional/Europe/United_Kingdom/Scotland/Dumfries_and_Galloway/Society_and_Culture/Religion")
    )
      .map {
        categories =>
          WebSiteCategoriesText("uri", "origUri", categories, "text")
      }.toDS()

    val actual: Array[Seq[String]] = truncateOrigCategoriesToTop3(sample).collect().map(_.categories)
    val expected = Seq(
      Seq("blerg"),
      Seq("blergo/blargo"),
      Seq("blergo/blargo"))
    actual should contain allElementsOf expected
  }

  it should "break into sentences depending on how common a category is" in {
    import spark.implicits._
    val text = "hours and one of our kind and friendly personal shoppers will help you navigate through our website, " +
      "help conduct advanced searches, help you choose the item you are looking for with the specifications you are seeking," +
      " read you the specifications of any item and consult with you about the products themselves. " +
      "There is no charge for the help of this personal shopper for any American with a disability. Finally, " +
      "your personal shopper will explain our Privacy Policy and Terms of Service, " +
      "and help you place an order if you so desire. Home About Us Location Rebate Center My Account Sign " +
      "InNew Customer? Start Here Your password and email"

    println(Text.splitAndClean(text).length)

    val sample = (Seq.fill(6){"religion"} ++ Seq.fill(2){"sport"} ++ Seq.fill(4){"computers"}).map{
      category =>
        WebSiteCategoriesText("uri1", "uri1", Seq(category), text)
    }.toDS

    val actual = breakIntoSentences(16, 2)(sample)

    actual
      .selectExpr("explode(categories) as category")
      .groupBy('category)
      .count
      .as[(String, Long)]
      .collect() should contain theSameElementsAs Seq(("religion", 6*7), ("computers", 4*32))
  }
}
