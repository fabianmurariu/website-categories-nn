package com.bytes32.prenn

import com.bytes32.prenn.PreNNProcessor.WebSiteCategoriesText
import org.apache.spark.sql.{Row, SparkSession}
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
      WebSiteFeature("uri1", "origUri1", Seq(5, 1, 2, 6, 5, 7), Seq(0, 1, 0)),
      WebSiteFeature("uri2", "origUri2", Seq(5, 4, 5, 1, 2, 0), Seq(1, 0, 0)),
      WebSiteFeature("uri3", "origUri3", Seq(5, 2, 0, 0, 0, 0), Seq(0, 0, 1)))
  }

  it should "create the embeddings matrix from glo vectors" in {
    import spark.implicits._
    val vocabulary = Map("the" -> 2, "cat" -> 1)
    val gloVectors = Seq(GloVector("the", Seq(1f, 2f, 3f)), GloVector("cat", Seq(3f, 2f, 1f))).toDS()
    val actual = PreNNTokenizer.generateEmbeddings(vocabulary, gloVectors).collect()

    actual should contain theSameElementsInOrderAs List(
      Row(Seq(0f, 0f, 0f)),
      Row(Seq(3f, 2f, 1f)),
      Row(Seq(1f, 2f, 3f)))
  }

}
