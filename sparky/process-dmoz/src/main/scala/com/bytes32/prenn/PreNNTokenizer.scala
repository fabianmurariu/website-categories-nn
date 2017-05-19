package com.bytes32.prenn

import com.bytes32.prenn.PreNNProcessor.WebSiteCategoriesText
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, StringIndexer, Tokenizer}
import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object PreNNTokenizer extends HasSpark with JobRunner with LazyLogging {

  def main(args: Array[String]): Unit = {
    val config = parse(args)

    implicit val spark = makeSparkSession("PreNNDataSetGenerator")
    import spark.implicits._

    val gloVectors = loadGloVectors(config.gloVectorsPath)
    val websitesCategoriesText = spark.read.parquet(config.websitesCleanPath).as[WebSiteCategoriesText]

    val (vocabulary: Map[String, Int], preFeatures) = getVocabularySplitTextIntoTokens(config.vocabSize, websitesCategoriesText)
    val embeddings = generateEmbeddings(vocabulary, gloVectors)

    val (features, labels) = featuresAndLabels(config.sequenceLength, vocabulary, preFeatures)
    features.write.json(config.outputPath + "/features")
    embeddings.write.json(config.outputPath + "/embeddings")
    labels.write.csv(config.outputPath + "/labels")
    vocabulary.toSeq.toDF("word", "id").write.json("/vocabulary")
  }

  def loadGloVectors(gloVectorsPath: String)(implicit spark: SparkSession): Dataset[GloVector] = {
    import spark.implicits._
    spark.read.text(gloVectorsPath).map {
      case Row(line: String) =>
        val tokens = line.split(" ")
        GloVector(tokens.head, tokens.tail.map(_.toFloat))
    }
  }

  def generateEmbeddings(vocabulary: Map[String, Int], gloVectors: Dataset[GloVector])
                        (implicit spark: SparkSession): DataFrame = {
    import spark.implicits._
    val gvSize = gloVectors.head().vector.length //too much?

    val zeroEmbedding = Seq(
      (0, Seq.fill(gvSize)(0f))
    ).toDF("id", "vector")

    val wordVectors = gloVectors.flatMap(gv => {
      val wordIndex = vocabulary.get(gv.word)
      wordIndex.map(id => id -> gv.vector)
    }).toDF("id", "vector")
      .sort("id")

    zeroEmbedding
      .union(wordVectors)
      .drop("id")
      .coalesce(1)
  }

  case class WebSiteFeature(uri: String, origUri: String, paddedWords: Seq[Int], category: Seq[Int])

  case class GloVector(word: String, vector: Seq[Float])

  def parse(args: Array[String]): Config = {
    val parser = new scopt.OptionParser[Config](getClass.getSimpleName) {
      opt[Unit]("local").action((_, config) =>
        config.copy(local = true)
      )

      opt[String]("vocabSize").optional().action((path, config) =>
        config.copy(vocabSize = path.toInt)).text("vocabulary size default 20000")
      opt[String]("sequenceLength").optional().action((path, config) =>
        config.copy(sequenceLength = path.toInt)).text("Size of a sentence default 1000")
      opt[String]("websitesCleanPath").required().action((path, config) =>
        config.copy(websitesCleanPath = path)).text("Path to output of categories and text")
      opt[String]("gloVectorsPath").required().action((path, config) =>
        config.copy(gloVectorsPath = path)).text("Path to word vectors")
      opt[String]("outputPath").required().action((path, config) =>
        config.copy(outputPath = path)).text("Path for the job output")

      override def reportError(msg: String): Unit = throw new IllegalArgumentException(s"$msg\n$usage")
    }

    parser.parse(args, Config(null, null, null)).get

  }

  case class Config(websitesCleanPath: String,
                    gloVectorsPath: String,
                    outputPath: String,
                    vocabSize: Int = 20000,
                    sequenceLength: Int = 1000, local: Boolean = false)

  def getVocabularySplitTextIntoTokens(vocabSize: Int, ds: Dataset[WebSiteCategoriesText])(implicit spark: SparkSession): (Map[String, Int], DataFrame) = {
    import spark.implicits._

    val tkz = new Tokenizer().setInputCol("text").setOutputCol("tokens")
    val tokens = tkz.transform(ds)

    val countVectorizer = new CountVectorizer()
      .setInputCol("tokens")
      .setVocabSize(vocabSize)
      .setOutputCol("counts")

    val model: CountVectorizerModel = countVectorizer.fit(tokens)
    val vocabulary = Array.tabulate(model.vocabulary.length) { i => model.vocabulary(i) -> i }.toMap
    val expandCategories = tokens.select('uri, 'origUri, explode('categories).as("category"), 'text, 'tokens)
    (vocabulary, expandCategories)
  }

  def featuresAndLabels(sequenceLength: Int, vocabulary: Map[String, Int], source: DataFrame)(implicit spark: SparkSession): (Dataset[WebSiteFeature], Dataset[String]) = {
    import spark.implicits._

    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(source)

    val features = source.map {
      case Row(uri: String, origUri: String, category: String, _, tokens: Seq[_]) =>
        val oheLabel = indexer.labels.map(l => if (l.equals(category)) 1 else 0)
        val numericTokens = tokens.view
          .map(word => vocabulary.getOrElse(word.toString, 0))
          .filter(_ != 0)
          .padTo(sequenceLength, 0)
          .take(sequenceLength)

        WebSiteFeature(uri,
          origUri,
          numericTokens,
          oheLabel)
    }
    (features, indexer.labels.seq.toDS().coalesce(1))
  }
}
