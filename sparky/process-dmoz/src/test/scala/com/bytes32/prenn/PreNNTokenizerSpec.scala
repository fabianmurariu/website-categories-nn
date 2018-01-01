package com.bytes32.prenn

import org.apache.spark.sql.{Row, SparkSession}
import org.scalacheck.Gen
import org.scalatest.{FlatSpec, Matchers}

/**
  * Created by murariuf on 18/05/2017.
  */
class PreNNTokenizerSpec extends FlatSpec with Matchers with HasSpark {

  implicit val spark: SparkSession = makeSparkSession("test", local = true)

  import PreNNTokenizer._

  "PreNNTokenizer" should "find a vocabulary for a corpus of text" in {
    import spark.implicits._
    val sample = Seq(
      WebSiteCategoriesText("uri1", "origUri1", Seq("sport"), "the quick brow fox dives over the lazy dog"),
      WebSiteCategoriesText("uri3", "origUri3", Seq("health"), "the fox is healthy")
    ).toDS()

    val (vocabulary, source) = getVocabularySplitTextIntoTokens(50, sample)
    vocabulary.keySet should contain theSameElementsAs Set("lazy", "is", "dives", "brow", "dog", "over", "healthy", "quick", "fox", "the")
  }

  it should "transform sentences and categories into numerical features" in {
    import spark.implicits._
    val sample = Seq(
      ("uri1", "origUri1", "sport", "text", "the quick brow fox dives over the lazy dog".split(" ")),
      ("uri2", "origUri2", "news", "text", "the dog notices the quick fox and wags his tail".split(" ")),
      ("uri3", "origUri3", "health", "text", "the fox is healthy".split(" "))
    ).toDF("uri", "origUri", "category", "text", "tokens")

    val vocabulary = Map("quick" -> 1, "fox" -> 2, "brown" -> 3, "dog" -> 4, "the" -> 5, "over" -> 6, "lazy" -> 7)
    val (features, labels) = featuresAndLabels(6, vocabulary, sample)

    labels.collect() should contain theSameElementsAs Seq("sport", "news", "health")

    features.collect() should contain theSameElementsAs Seq(
      WebSiteFeature("uri1", "origUri1", Seq(5, 1, 2, 6, 5, 7), Seq(0, 1, 0), "sport"),
      WebSiteFeature("uri2", "origUri2", Seq(5, 4, 5, 1, 2, 0), Seq(1, 0, 0), "news"),
      WebSiteFeature("uri3", "origUri3", Seq(5, 2, 0, 0, 0, 0), Seq(0, 0, 1), "health"))
  }

  it should "create the embeddings matrix from glo vectors" in {
    import spark.implicits._
    val vocabulary = Map("the" -> 2, "cat" -> 1, "no-embeddings" -> 3)
    val gloVectors = Seq(
      GloVector("the", Seq(1f, 2f, 3f)),
      GloVector("cat", Seq(3f, 2f, 1f)),
      GloVector("no-vocab", Seq(4f, 5f, 6f))).toDS()
    val (actual, vocabWithEmbeddings) = PreNNTokenizer.generateEmbeddings(vocabulary, gloVectors)

    vocabWithEmbeddings should be(Map("the" -> 2, "cat" -> 1, "no-embeddings" -> 3))

    actual.collect() should contain theSameElementsInOrderAs List(
      Row(Seq(3f, 2f, 1f)),
      Row(Seq(1f, 2f, 3f)),
      Row(Seq(0f, 0f, 0f))
    )
  }

  it should "create class weights " in {
    import spark.implicits._
    val sample = Seq(
      WebSiteCategoriesText("uri1", "origUri1", Seq("sport"), ""),
      WebSiteCategoriesText("uri3", "origUri3", Seq("sport"), ""),
      WebSiteCategoriesText("uri3", "origUri3", Seq("health"), ""),
      WebSiteCategoriesText("uri3", "origUri3", Seq("health"), ""),
      WebSiteCategoriesText("uri3", "origUri3", Seq("health"), ""),
      WebSiteCategoriesText("uri3", "origUri3", Seq("health"), ""),
      WebSiteCategoriesText("uri3", "origUri3", Seq("health"), ""),
      WebSiteCategoriesText("uri3", "origUri3", Seq("technology"), "")
    ).toDS()

    PreNNTokenizer.classWeights(sample) should be(Map("sport" -> 5f / 2, "health" -> 1f, "technology" -> 5f / 1))
  }


  it should "split the dataset into training, test and validation" in {
    import spark.implicits._
    val sample = sampleGen(genFeatures).take(10000).toVector.toDS()

    val sets = PreNNTokenizer.splitTrainTestValid(sample)
    val trainCount = sets("train").count()
    val testCount = sets("test").count()
    val validCount = sets("valid").count()

    trainCount should === (8000L +- 800L)
    testCount should be (1000L +- 100)
    validCount should be (1000L +- 100)

    sets("train").collect().toSet.intersect(sets("test").collect().toSet) shouldBe empty
    sets("train").collect().toSet.intersect(sets("valid").collect().toSet) shouldBe empty
    sets("test").collect().toSet.intersect(sets("valid").collect().toSet) shouldBe empty
  }

  lazy val genFeatures: Gen[WebSiteFeature] = for {
    uri <- Gen.alphaNumStr
    origUri <- Gen.alphaNumStr
    categoryName <- Gen.alphaNumStr
    features <- Gen.buildableOfN[Seq[Int], Int](10, Gen.choose(0, 50))
    cats: Seq[Int] <- Gen.oneOf(List(Seq(0, 1), Seq(1, 0)))
  } yield WebSiteFeature(uri, origUri, features, cats, categoryName)

  def sampleGen[T](g: Gen[T], failBudget: Int = 10): Stream[T] = g.sample match {
    case Some(t) => t #:: sampleGen(g, failBudget)
    case None if failBudget > 0 => sampleGen(g, failBudget - 1)
    case None => Stream.empty[T]
  }
}
