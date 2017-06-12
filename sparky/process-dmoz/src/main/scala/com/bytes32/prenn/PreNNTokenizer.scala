package com.bytes32.prenn

import com.bytes32.prenn.PreNNProcessor.WebSiteCategoriesText
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, StringIndexer, Tokenizer}
import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.util.Random

object PreNNTokenizer extends HasSpark with JobRunner with LazyLogging {

  type Vocabulary = Map[String, Int]

  def main(args: Array[String]): Unit = {
    val config = parse(args)

    implicit val spark = makeSparkSession("PreNNDataSetGenerator")
    import spark.implicits._

    val gloVectors = loadGloVectors(config.gloVectorsPath)
    val websitesCategoriesText = spark.read.parquet(config.websitesCleanPath).as[WebSiteCategoriesText]

    val (vocabulary: Map[String, Int], preFeatures) = getVocabularySplitTextIntoTokens(config.vocabSize, websitesCategoriesText)
    val (embeddings, vocabWithEmbeddings: Vocabulary) = generateEmbeddings(vocabulary, gloVectors)


    val (features, labels) = featuresAndLabels(config.sequenceLength, vocabWithEmbeddings, balanceDatSetsByCategory(preFeatures))
    val featuresPath = config.outputPath + "/features"

    runForOutput(featuresPath) {
      features.write.json(featuresPath)
    }

    val embeddingsPath = config.outputPath + "/embeddings"
    runForOutput(embeddingsPath) {
      embeddings.write.json(embeddingsPath)
    }

    val labelsPath = config.outputPath + "/labels"
    runForOutput(labelsPath) {
      labels.write.csv(labelsPath)
    }

    val vocabularyPath = config.outputPath + "/vocabulary"
    runForOutput(vocabularyPath) {
      vocabWithEmbeddings.toSeq.toDF("word", "id").repartition(1).write.json(vocabularyPath)
    }
  }

  def balanceDatSetsByCategory(features: DataFrame)(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._
    val categoryCounts = features.groupBy('category).count()
    categoryCounts.show(150)
    val counts = categoryCounts.collect().map { case Row(category: String, count: Long) => category -> count }
    val min = counts.map(_._2).min
    val weighted = counts.map { case (name, count) => name -> (min.toDouble / count) }.toMap
    features.stat.sampleBy("category", weighted, Random.nextLong())
  }

  def loadGloVectors(gloVectorsPath: String)(implicit spark: SparkSession): Dataset[GloVector] = {
    import spark.implicits._
    spark.read.text(gloVectorsPath).map {
      case Row(line: String) =>
        val tokens = line.split(" ")
        GloVector(tokens.head, tokens.tail.map(_.toFloat))
    }
  }

  def generateEmbeddings(vocabulary: Vocabulary, gloVectors: Dataset[GloVector])
                        (implicit spark: SparkSession): (DataFrame, Vocabulary) = {
    import spark.implicits._
    val gvSize = gloVectors.head().vector.length //too much?

    val zeroEmbedding = Seq(
      Seq.fill(gvSize)(0f)
    ).toDF("vector")

    /* get embeddings only for words in vocabulary */
    val wordVectors = gloVectors.select('word).map(_.getString(0)).collect().toSet
    val newVocabulary = wordVectors.intersect(vocabulary.keySet).zipWithIndex.map {
      case (word, i: Int) => word -> (i + 1)
    }.toMap // all the words for which we have embeddings

    val wordEmbeddings = gloVectors.flatMap { glV =>
      newVocabulary.get(glV.word).map(id => id -> glV.vector)
    }.toDF("id", "vector")
      .sort("id")
      .drop("id")

    val embeddings = zeroEmbedding
      .drop("id", "word")
      .union(wordEmbeddings)
      .coalesce(1)
    (embeddings, newVocabulary)
  }

  case class WebSiteFeature(uri: String, origUri: String, paddedWords: Seq[Int], category: Seq[Int])

  case class GloVector(word: String, vector: Seq[Float])

  def parse(args: Array[String]): Config = {
    val parser = new scopt.OptionParser[Config](getClass.getSimpleName) {
      opt[Unit]("local").action((_, config) =>
        config.copy(local = true)
      )

      opt[String]('v',"vocabSize").optional().action((path, config) =>
        config.copy(vocabSize = path.toInt)).text("vocabulary size default 20000")
      opt[String]('s',"sequenceLength").optional().action((path, config) =>
        config.copy(sequenceLength = path.toInt)).text("Size of a sentence default 1000")
      opt[String]('w',"websitesCleanPath").required().action((path, config) =>
        config.copy(websitesCleanPath = path)).text("Path to output of categories and text")
      opt[String]('g',"gloVectorsPath").required().action((path, config) =>
        config.copy(gloVectorsPath = path)).text("Path to word vectors")
      opt[String]('o',"outputPath").required().action((path, config) =>
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

  def getVocabularySplitTextIntoTokens(vocabSize: Int, ds: Dataset[WebSiteCategoriesText])
                                      (implicit spark: SparkSession): (Vocabulary, DataFrame) = {
    import org.apache.spark.sql.functions.udf
    import spark.implicits._

    val processText = udf {
      Text.splitAndClean
    }

    val tokens = ds.withColumn("tokens", processText('text))

    val countVectorizer = new CountVectorizer()
      .setInputCol("tokens")
      .setVocabSize(vocabSize)
      .setOutputCol("counts")

    val model: CountVectorizerModel = countVectorizer.fit(tokens)
    val vocabulary = Array.tabulate(model.vocabulary.length) { i => model.vocabulary(i) -> i }.toMap
    val expandCategories = tokens.select('uri, 'origUri, explode('categories).as("category"), 'text, 'tokens)
    (vocabulary, expandCategories)
  }

  def featuresAndLabels(sequenceLength: Int, vocabulary: Vocabulary, source: DataFrame)(implicit spark: SparkSession): (Dataset[WebSiteFeature], Dataset[String]) = {
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
