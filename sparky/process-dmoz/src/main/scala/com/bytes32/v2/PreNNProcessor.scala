package com.bytes32.v2

import com.bytes32.prenn._
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.{Dataset, SparkSession}

object PreNNProcessor extends HasSpark with JobRunner with LazyLogging with ProcessorConfig with TextCleanup {

  def truncateOrigCategoriesToTop3(cats: Dataset[WebSiteCategoriesText])
                                  (implicit spark: SparkSession): Dataset[WebSiteCategoriesText] = {
    import spark.implicits._

    def makeCategory(origCategory: String, sep: String = "/"): String = {
      origCategory
        .toLowerCase()
        .split("\\/")
        .take(4)
        .mkString(sep)
        .stripPrefix(s"top$sep")
    }

    cats.flatMap {
      case WebSiteCategoriesText(uri, origUri, categories, text, origCategories) =>
        categories.map(makeCategory(_)).map {
          newCat =>
            WebSiteCategoriesText(uri, origUri, Seq(newCat), text, origCategories)
        }
    }.filter { ws =>
      ws.categories.nonEmpty && (
        !ws.categories.head.contains("regional") && !ws.categories.head.contains("world"))
    }
  }

  def breakIntoSentences(sentenceLength: Int, topCategories: Int = 300)
                        (cats: Dataset[WebSiteCategoriesText])
                        (implicit spark: SparkSession): Dataset[WebSiteCategoriesText] = {
    import spark.implicits._

    val categoriesCounts: Map[String, Long] = cats
      .selectExpr("explode(categories) as category")
      .groupBy('category)
      .count()
      .orderBy('count.desc)
      .where('count > topCategories) // at least 300 texts
      .as[(String, Long)]
      .collect()
      .toMap

    println("WTF!" + categoriesCounts)

    val (_,minTextCount) = categoriesCounts.minBy(_._2)
    val (_,maxTextCount) = categoriesCounts.maxBy(_._2)

    println("WTF2!", minTextCount, maxTextCount)

    // define a sliding window over a text depending how popular a category is
    // a very popular category moves the sliding window faster
    // an un-popular category moves it slower sampling more of the text

    def slidingWindowStep(min:Long, max:Long, categoryCount:Long):Int= {
      //define a line from 1 to sentenceLength
      //we slide with a step of sentenceLength for the most common category
      //we slide with a step of 1 for the least common category
      //we figure out a line and pick numbers on it for everything in between
      val a: Double = (sentenceLength.toDouble - 1d) / (max.toDouble - min.toDouble)
      val b: Double = 1d - (min * a)
      math.max(1, (a * categoryCount + b).toInt)
    }

    categoriesCounts.foreach{
      case (_, c) =>
        println(slidingWindowStep(minTextCount, maxTextCount, c))
    }

    cats.flatMap {
      case WebSiteCategoriesText(uri, origUri, categories, text: String, origCategories) =>
        if (categoriesCounts.contains(categories.head)){
          val categoryCount = categoriesCounts(categories.head)
          val stepLength = slidingWindowStep(minTextCount, maxTextCount, categoryCount)
          Text.splitAndClean(text)
            .sliding(sentenceLength, stepLength).map {
            tks => WebSiteCategoriesText(uri, origUri, categories, tks.mkString(" "), origCategories)
          }
        } else Seq.empty
    }
  }

  def termFreqIdf(ds:Dataset[WebSiteCategoriesText], dictionarySize:Int=10000)
                 (implicit spark:SparkSession): Unit = {
    import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
    val docs = ds.selectExpr("explode(categories) as category", "text")

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val words = tokenizer
      .transform(docs)

    val hashingTF = new HashingTF()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .setNumFeatures(dictionarySize)

    val featurizedData =
      hashingTF.transform(words)

    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")

    val idfModel = idf.fit(featurizedData)

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

      (truncateOrigCategoriesToTop3 _ andThen breakIntoSentences(128)) (dmozTextCats)
        .write
        .option("compression", "snappy")
        .parquet(websitesCleanOutput)
    }
  }

}
