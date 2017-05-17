package com.bytes32.prenn

import com.optimaize.langdetect.i18n.LdLocale
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

  it should "load subcategories.jl as Seq[FilterCategory]" in {
    val cats = getClass.getResource("/subcategories.jl")
    val categories = loadCategories(cats.getPath)
    categories should contain atLeastOneOf(
      FilterCategory("science", Set("Science"), Set("Science/Technology")),
      FilterCategory("bikes", Set("Sports/Cycling"), Set.empty)
    )
  }

  it should "filter categories considering excludes" in {
    import spark.implicits._
    val ws = Seq(WebSiteCategoriesText("uri1", "origUri1", Seq("Top/Science/Technology"), "text1")).toDS()
    val filters = Seq(
      FilterCategory("science", Set("Science"), Set("Science/Technology")),
      FilterCategory("technology", Set("Science/Technology"), Set.empty)
    )
    val expected = Seq(WebSiteCategoriesText("uri1", "origUri1", Seq("technology"), "text1"))
    val actual = filterAndExpandWebSites(filters)(ws).collect()

    actual should contain theSameElementsAs expected
  }

  it should "filter and expand categories when there is one match" in {
    import spark.implicits._
    val ws = Seq(WebSiteCategoriesText("uri1", "origUri1", Seq("TV/Sports", "Sports", "Lifestyle"), "text1")).toDS()
    val filters = Seq(FilterCategory("sport", Set("TV/Sports"), Set.empty))
    val expected = Seq(WebSiteCategoriesText("uri1", "origUri1", Seq("sport"), "text1"))
    val actual = filterAndExpandWebSites(filters)(ws).collect()

    actual should contain theSameElementsAs expected
  }

  it should "filter and expand categories when there are 2 matches" in {
    import spark.implicits._
    val ws = Seq(WebSiteCategoriesText("uri1", "origUri1", Seq("TV/Sports", "Sports", "Lifestyle"), "text1")).toDS()
    val filters = Seq(
      FilterCategory("sport", Set("TV/Sports"), Set.empty),
      FilterCategory("lifestyle", Set("Lifestyle"), Set.empty)
    )
    val expected = Seq(
      WebSiteCategoriesText("uri1", "origUri1", Seq("sport"), "text1"),
      WebSiteCategoriesText("uri1", "origUri1", Seq("lifestyle"), "text1")
    )

    val actual = filterAndExpandWebSites(filters)(ws).collect()

    actual should contain theSameElementsAs expected

  }

  it should "detect english websites" in {
    import spark.implicits._
    val samples =
      Seq(WebSiteCategoriesText("uri1", "origUri1", Seq("general", "french"), "Le jour est brillant et nous avons des journaux à recevoir"),
        WebSiteCategoriesText("uri2", "origUri2", Seq("fonts"), "the quick brown fox jumps over the lazy ass dog")).toDS

    val actual = excludeNonEnglishWebsitesFlatten(samples).collect()
    actual should contain theSameElementsAs List(
      WebSiteCategoriesText("uri2", "origUri2", Seq("fonts"), "the quick brown fox jumps over the lazy ass dog"))
  }

  it should "not take too long to detect language" in {
    Language.detectEnglish("the quick brow fox jumped over the lazy dog") should be(Some(LdLocale.fromString("en")))
  }

}
