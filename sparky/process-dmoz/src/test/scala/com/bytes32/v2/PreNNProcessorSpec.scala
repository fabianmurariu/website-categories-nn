package com.bytes32.v2

import com.bytes32.prenn.{DMOZCats, HasSpark, WebSiteCategoriesText}
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
      Seq("blergo/blargo/the"))
    actual should contain allElementsOf expected
  }
}
