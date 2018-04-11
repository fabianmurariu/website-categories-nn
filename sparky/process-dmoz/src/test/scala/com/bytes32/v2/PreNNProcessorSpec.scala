package com.bytes32.v2

import com.bytes32.prenn._
import org.apache.spark.sql.{Dataset, SparkSession}
import org.scalatest.{FlatSpec, Matchers}

class PreNNProcessorSpec extends FlatSpec with Matchers with HasSpark {

  implicit val spark: SparkSession = makeSparkSession("test", local = true)

  import PreNNProcessor._

  "PreNNProcessor v2" should "truncate the original categories into the top 3 levels" in {
    import spark.implicits._
    val sample = Seq(
      "Top/Blerg",
      "Top/",
      "Top/Blergo/Blargo",
      "Top/Blergo/Blargo/The/Third",
      "Top/Regional/Europe/United_Kingdom/Scotland/Dumfries_and_Galloway/Society_and_Culture/Religion"
    )
      .map {
        category =>
          WebSiteCategory("uri", category, "text")
      }.toDS()

    val actual = removeRemainingWorldRegionalCatsLowercase(sample).collect().map(_.category)
    val expected = Seq(
      "blerg",
      "blergo/blargo",
      "blergo/blargo")
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
        WebSiteCategory("uri1", category, text)
    }.toDS

    val actual = breakIntoSentences(16, 2)(sample)

    actual
      .groupBy('category)
      .count
      .as[(String, Long)]
      .collect() should contain theSameElementsAs Seq(("religion", 6*7), ("computers", 4*24))
  }

  it should "expand categories based on counts stopping at cutoff" in {
    import spark.implicits._

    val sample = Seq(
      WebSiteCategory("ou1", "a/b/c","t1"),
      WebSiteCategory("ou1", "a/b/c1","t1"),
      WebSiteCategory("ou1", "a/b/c2","t1"),
      WebSiteCategory("ou1", "a/b/c","t1"),
      WebSiteCategory("ou1", "a1/b1","t1"),
      WebSiteCategory("ou1", "a1/b2","t1"),
      WebSiteCategory("ou1", "a2/b2/c2/c3","t1"),
      WebSiteCategory("ou1", "a2/b2/c2/c3","t1"),
      WebSiteCategory("ou1", "a2/b2/c2/c3","t1"),
      WebSiteCategory("ou1", "a2/b2/c2/c3","t1"),
      WebSiteCategory("ou1", "a2/b2/c2/c3","t1")
    )

    val expected = Seq(
      WebSiteCategory("u1", "a/b/other","t1"),
      WebSiteCategory("u1", "a/b/other","t1"),
      WebSiteCategory("u1", "a/b/other","t1"),
      WebSiteCategory("u1", "a/b/other","t1"),
      WebSiteCategory("u1", "a2/b2/c2","t1"),
      WebSiteCategory("u1", "a2/b2/c2","t1"),
      WebSiteCategory("u1", "a2/b2/c2","t1"),
      WebSiteCategory("u1", "a2/b2/c2","t1"),
      WebSiteCategory("u1", "a2/b2/c2","t1"),
      WebSiteCategory("u1", "a1/other","t1"),
      WebSiteCategory("u1", "a1/other","t1")
    )

    val actual = expandPopularCategories(3, 3)(sample.toDS()).collect()

    actual.diff(expected) should be(Seq.empty[WebSiteCategory])
  }
}
