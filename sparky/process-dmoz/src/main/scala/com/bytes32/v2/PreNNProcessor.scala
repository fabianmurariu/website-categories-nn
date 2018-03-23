package com.bytes32.v2

import com.bytes32.prenn._
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object PreNNProcessor extends HasSpark with JobRunner with LazyLogging with ProcessorConfig with TextCleanup {

  def expandPopularCategories(cutoff:Int = 2000, minArticles:Int = 100)
                             (websites: Dataset[WebSiteCategoriesText])
                             (implicit spark: SparkSession):Dataset[WebSiteCategoriesText] = {
    import org.apache.spark.sql.functions._
    import spark.implicits._

    val pickCategory = udf[String, String, Int] {
      (cat, n) => cat.toLowerCase.stripPrefix("top/").split("\\/").take(n).mkString("/")
    }

    def loop(ds: DataFrame, i:Int, cutoff:Int):DataFrame = {
      /* uri, category, text */
      val catCandidates = ds
        .where("category not like '%World%'").where("category not like '%Regional%'")
        .withColumn("cat", pickCategory('category, lit(i)))

      val counts = catCandidates
        .groupBy('cat).count().orderBy('count.desc)
        .cache()

      val categoryDepthCutoff = size(split(col("cat"),"\\/"))

      val showExpandedCatCounts = counts
        .where($"count" > cutoff)
        .where(categoryDepthCutoff <= 2)
        .withColumnRenamed("cat", "cat_selected")

      val expandCatCounts = showExpandedCatCounts
        .drop("count")

      val stopExpandCatCounts = counts
        .where($"count" <= cutoff or categoryDepthCutoff > 2)
        .withColumnRenamed("cat", "cat_remain")
        .drop("count")

      showExpandedCatCounts.show(150, truncate=false)

      if (expandCatCounts.collect().length <= 1) {
        /* either all the cats are below the cutoff or we only have 1 contender remaining */
        catCandidates
      } else {
        val expandWebsites = catCandidates
          .join(expandCatCounts, $"cat" === $"cat_selected", "inner")
          .drop("cat_selected")
          .drop("cat")

        val stopExpandWebsites = catCandidates
          .join(stopExpandCatCounts, $"cat" === $"cat_remain", "inner")
          .drop("cat_remain")

        println("WTF -> ", catCandidates.count(), expandWebsites.count(), stopExpandWebsites.count())

        loop(expandWebsites.persist(), i+1, cutoff).union(stopExpandWebsites)
      }
    }
    val slimWebsites = websites.selectExpr("uri", "explode(categories) as category", "text")
    val results = loop(slimWebsites, 2, cutoff).cache()
    val resCounts = results.groupBy('cat).count().orderBy('count.desc).withColumnRenamed("cat", "cat0")
    results
      .join(resCounts, $"cat" === $"cat0")
      .drop("cat0")
      .where("count > 5")
      .map{
        case Row(uri:String, category:String, text:String, cat:String, count:Long) =>
          val catParts = cat.split("\\/")
          if (catParts.length == 1 && count < minArticles) {
            WebSiteCategoriesText(uri, uri, Seq(cat + "/other"), text)
          } else if (catParts.length > 1 && count < minArticles) {
            val newCat = (catParts.dropRight(1) ++ Array(s"other")).mkString("/")
            WebSiteCategoriesText(uri, uri, Seq(newCat), text)
          } else {
            WebSiteCategoriesText(uri, uri, Seq(cat), text)
          }
      }
  }

  def removeRemainingWorldRegionalCatsLowercase(cats: Dataset[WebSiteCategoriesText])
                                               (implicit spark: SparkSession): Dataset[WebSiteCategoriesText] = {
    import spark.implicits._

    def makeCategory(origCategory: String, sep: String = "/"): String = {
      origCategory
        .toLowerCase()
        .split("\\/")
        .mkString(sep)
        .stripPrefix(s"top$sep")
        .stripPrefix("top")
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
      val minStep = 4d
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

  def termFreqIdf(vocabularySize: Int = 30000)(ds: Dataset[WebSiteCategoriesText])
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
            .filter(word => docFreqIndex.contains(word) && docFreqIndex(word) > 1)
            .mkString(" ")
          WebSiteCategoriesText(uri, "", Seq(category), newText)
      }
  }

  def recoverRegionalAndWorldCategories(ds: Dataset[WebSiteCategoriesText])
                                       (implicit spark: SparkSession): Dataset[WebSiteCategoriesText] = {
    import spark.implicits._
    import org.apache.spark.sql.functions._
    val worldColFilter = 'category.like("%/Regional%") or 'category.like("%/World%")

    val worldDf = ds.select('origUri.as("world_uri"), explode('categories).as("category"))
      .where(worldColFilter)

    val restDf = ds.select('origUri.as("rest_uri"), explode('categories).as("category"))
      .where(not(worldColFilter))
      .withColumnRenamed("category", "substitute_category")

    val substitituteMap = worldDf.join(restDf, $"world_uri" === $"rest_uri")
      .select('category, 'substitute_category)
      .map { case Row(category: String, substituteCategory: String) => category -> substituteCategory }
      .collect()
      .groupBy(_._1)
      .map {
        case (category, substitutes) =>
          val (substitute, _) =
            substitutes.map(_._2)
              .groupBy(identity)
              .map { case (k, v) => k -> v.length }
              .maxBy(_._2)
          category -> substitute
      }
    ds.map {
      ws =>
        val substitutes = ws.categories.map(cat => substitituteMap.getOrElse(cat, cat))
        ws.copy(categories = substitutes)
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

      (recoverRegionalAndWorldCategories _ andThen
        removeRemainingWorldRegionalCatsLowercase andThen
        expandPopularCategories() andThen
        termFreqIdf() andThen
        breakIntoSentences(128)) (dmozTextCats)
        .write
        .option("compression", "snappy")
        .parquet(websitesCleanOutput)
    }
  }

}
