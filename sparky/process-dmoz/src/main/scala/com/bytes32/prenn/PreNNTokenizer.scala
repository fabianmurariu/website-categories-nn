package com.bytes32.prenn

import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, StringIndexer}
import org.apache.spark.sql._
import org.apache.spark.sql.functions.explode

object PreNNTokenizer extends HasSpark with JobRunner with LazyLogging {

  def splitTrainTestValid[T](features: Dataset[T], buckets: Array[Double] = Array(0, 0.1, 0.2, 1), labels: Seq[String] = Seq("valid", "test", "train"))
                            (implicit spark: SparkSession): Map[String, Dataset[T]] = {

    val weights: Array[Double] = (buckets zip buckets.drop(1)).map { case (start, end) => end - start }

    (labels zip features.randomSplit(weights)).toMap
  }


  type Vocabulary = Map[String, Int]

  def main(args: Array[String]): Unit = {
    val config = parse(args)

    implicit val spark: SparkSession = makeSparkSession("PreNNDataSetGenerator", config.local)
    import spark.implicits._

    val gloVectors = loadGloVectors(config.gloVectorsPath)
    val websitesCategoriesText = spark.read.parquet(config.websitesCleanPath).as[WebSiteCategoriesText]

    val classWeightsPath = config.outputPath + "/class-weights"
    val (classWeightsInner, classCountsInner) = classWeights(websitesCategoriesText)
    runForOutput(classWeightsPath) {
      val classWeightsDF = Seq("classWeights" -> classWeightsInner).toDF("name", "weights").coalesce(1)
      classWeightsDF.write.json(classWeightsPath)
    }

    val (vocabulary: Map[String, Int], preFeatures) = getVocabularySplitTextIntoTokens(config.vocabSize, websitesCategoriesText)
    val (embeddings, vocabWithEmbeddings: Vocabulary) = generateEmbeddings(vocabulary, gloVectors)

    val (features, labels) = featuresAndLabels(config.sequenceLength, vocabWithEmbeddings, preFeatures)
    val featuresPath = config.outputPath + "/features"

    features.cache()

    val featSplit = splitTrainTestValid(features)
    val trainPath = featuresPath + "/train"
    val testPath = featuresPath + "/test"
    val validPath = featuresPath + "/valid"

    runForOutput(trainPath, testPath, validPath) {
      featSplit("train").write.option("compression", "gzip").json(trainPath)
      featSplit("test").write.option("compression", "gzip").json(testPath)
      featSplit("valid").write.option("compression", "gzip").json(validPath)
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

  def classWeights(ds: Dataset[WebSiteCategoriesText])(implicit spark: SparkSession): (Map[String, Double], Map[String, Long]) = {
    import spark.implicits._
    val categoryCount: Map[String, Long] = ds
      .select("categories")
      .selectExpr("explode(categories) as category")
      .groupBy("category")
      .count()
      .map { case Row(category: String, count: Long) => category -> count }
      .collect()
      .toMap

    val max = categoryCount.values.max.toDouble

    val classWeights = categoryCount.map { case (cat, count) => cat -> max / count }
    (classWeights, categoryCount)
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

    val zeroEmbedding = Seq.fill(gvSize)(0f)

    /* get embeddings only for words in vocabulary */
    val newVocabulary = vocabulary.toSeq.toDS()

    val wordEmbeddings = newVocabulary.joinWith(gloVectors, $"_1" === $"word", "left_outer")
      .map {
        case ((word, id), GloVector(_, vector)) => (word, id, vector)
        case ((word, id), null) => (word, id, zeroEmbedding)
      }.toDF("word", "id", "vector")
      .sort("id")
      .drop("word", "id")
      .coalesce(1)

    (wordEmbeddings, vocabulary)
  }

  case class WebSiteFeature(uri: String, origUri: String, paddedWords: Seq[Int], category: Seq[Int], categoryName:String)

  case class GloVector(word: String, vector: Seq[Float])

  def parse(args: Array[String]): Config = {
    val parser = new scopt.OptionParser[Config](getClass.getSimpleName) {
      opt[Unit]("local").action((_, config) =>
        config.copy(local = true)
      )

      opt[String]('v', "vocabSize").optional().action((path, config) =>
        config.copy(vocabSize = path.toInt)).text("vocabulary size default 20000")
      opt[String]('s', "sequenceLength").optional().action((path, config) =>
        config.copy(sequenceLength = path.toInt)).text("Size of a sentence default 1000")
      opt[String]('w', "websitesCleanPath").required().action((path, config) =>
        config.copy(websitesCleanPath = path)).text("Path to output of categories and text")
      opt[String]('g', "gloVectorsPath").required().action((path, config) =>
        config.copy(gloVectorsPath = path)).text("Path to word vectors")
      opt[String]('o', "outputPath").required().action((path, config) =>
        config.copy(outputPath = path)).text("Path for the job output")

      override def reportError(msg: String): Unit = throw new IllegalArgumentException(s"$msg\n$usage")
    }

    parser.parse(args, Config(null, null, null)).get

  }

  case class Config(websitesCleanPath: String,
                    gloVectorsPath: String,
                    outputPath: String,
                    vocabSize: Int = 20000,
                    sequenceLength: Int = 128, local: Boolean = false)

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

  def featuresAndLabels(sequenceLength: Int, vocabulary: Vocabulary, source: DataFrame)
                       (implicit spark: SparkSession): (Dataset[WebSiteFeature], Dataset[String]) = {
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
          oheLabel,
          category)
    }
    (features, indexer.labels.seq.toDS().coalesce(1))
  }
}
