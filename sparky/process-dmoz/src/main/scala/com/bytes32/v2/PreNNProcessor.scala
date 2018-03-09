package com.bytes32.v2

import com.bytes32.prenn._
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.{Dataset, Row, SparkSession}

object PreNNProcessor extends HasSpark with JobRunner with LazyLogging with ProcessorConfig with TextCleanup {

  def truncateOrigCategoriesToTop3(cats: Dataset[WebSiteCategoriesText])
                                  (implicit spark: SparkSession): Dataset[WebSiteCategoriesText] = {
    import spark.implicits._

    def makeCategory(origCategory: String, sep: String = "/", depth:Int=3): String = {
      origCategory
        .toLowerCase()
        .split("\\/")
        .take(depth)
        .mkString(sep)
        .stripPrefix(s"top$sep")
    }

    cats
      .map(ws => ws.copy(text = Text.splitAndClean(ws.text).mkString(" ")))
      .flatMap {
        case WebSiteCategoriesText(uri, origUri, categories, text, origCategories) =>
          categories.map(makeCategory(_)).map {
            newCat =>
              WebSiteCategoriesText(uri, origUri, Seq(newCat), text, origCategories)
          }
      }.filter { ws =>
      ws.categories.nonEmpty && (
        !ws.categories.head.contains("regional") &&
          !ws.categories.head.contains("world") &&
        !ws.categories.head.startsWith("top"))
    }
  }

  def breakIntoSentences(sentenceLength: Int, maxDocumentCountPerCat: Int = 50)
                        (cats: Dataset[WebSiteCategoriesText])
                        (implicit spark: SparkSession): Dataset[WebSiteCategoriesText] = {
    import spark.implicits._

    val categoriesCounts: Map[String, Long] = cats
      .selectExpr("explode(categories) as category")
      .groupBy('category)
      .count()
      .orderBy('count.desc)
      .where('count > maxDocumentCountPerCat) // at least 300 texts
      .as[(String, Long)]
      .collect()
      .toMap

    val (_, minTextCount) = categoriesCounts.minBy(_._2)
    val (_, maxTextCount) = categoriesCounts.maxBy(_._2)

    // define a sliding window over a text depending how popular a category is
    // a very popular category moves the sliding window faster
    // an un-popular category moves it slower sampling more of the text

    def slidingWindowStep(min: Long, max: Long, categoryCount: Long): Int = {
      //define a line from 1 to sentenceLength
      //we slide with a step of sentenceLength for the most common category
      //we slide with a step of 1 for the least common category
      //we figure out a line and pick numbers on it for everything in between
      val minStep = 3d
      val a: Double = (sentenceLength.toDouble - minStep) / (max.toDouble - min.toDouble)
      val b: Double = minStep - (min * a)
      math.max(1, (a * categoryCount + b).toInt)
    }

    cats.flatMap {
      case WebSiteCategoriesText(uri, origUri, categories, text: String, origCategories) =>
        if (categoriesCounts.contains(categories.head)) {
          val categoryCount = categoriesCounts(categories.head)
          val stepLength = slidingWindowStep(minTextCount, maxTextCount, categoryCount)
          text.split("\\s+")
            .sliding(sentenceLength, stepLength).map {
            tks => WebSiteCategoriesText(uri, origUri, categories, tks.mkString(" "), origCategories)
          }
        } else Seq.empty
    }
  }

  def termFreqIdf(vocabularySize: Int = 20000)(ds: Dataset[WebSiteCategoriesText])
                 (implicit spark: SparkSession): Dataset[WebSiteCategoriesText] = {
    import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, CountVectorizer, CountVectorizerModel}
    import spark.implicits._
    val docs = ds.selectExpr("uri", "explode(categories) as category", "text")

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val words = tokenizer
      .transform(docs)

    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .setVocabSize(vocabularySize)
      .setMinDF(1)
      .fit(words)

    val featurizedData =
      cvModel.transform(words)

    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")

    val idfModel = idf.fit(featurizedData)
    idfModel
      .transform(featurizedData)
      .select("uri", "category", "text", "features")
      .map {
        case Row(uri: String, category: String, text: String, features: org.apache.spark.ml.linalg.SparseVector) =>
          val docFreqIndex = features.indices.view.zipWithIndex
            .map { case (wordIndex, freqIndex) =>
              val word = cvModel.vocabulary(wordIndex)
              val freq = features.values(freqIndex)
              word -> freq
            }.toMap
          val newText = text
            .split("\\s+")
            .filter(word => docFreqIndex.contains(word ) && docFreqIndex(word) > 1)
            .mkString(" ")
          WebSiteCategoriesText(uri, "", Seq(category), newText)
      }
  }

  def main(args: Array[String]): Unit = {
    val Config(websitesRawInput, websitesTextOutput, _, websitesCleanOutput, local) = parseArgs(args)

    implicit val spark: SparkSession = makeSparkSession("PreNNProcessor", local)
    import spark.implicits._

    /* filter non-english websites */
    runForOutput(websitesTextOutput) {
      val rawHtml = spark.read.json(websitesRawInput)
      (extractTextFromRawHtmlWithCategories _ andThen excludeNonEnglishWebsites) (rawHtml)
        .write
        .option("compression", "snappy")
        .parquet(websitesTextOutput)
    }

    runForOutput(websitesCleanOutput) {
      val dmozTextCats = spark.read
        .parquet(websitesTextOutput)
        .as[WebSiteCategoriesText]

      (truncateOrigCategoriesToTop3 _ andThen termFreqIdf() andThen breakIntoSentences(128)) (dmozTextCats)
        .write
        .option("compression", "snappy")
        .parquet(websitesCleanOutput)
    }
  }

}
