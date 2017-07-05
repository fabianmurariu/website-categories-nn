package com.bytes32.serve

import java.net.URI

import com.twitter.util.Future
import org.jsoup.Jsoup

/**
  * Created by murariuf on 09/06/2017.
  */
class PredictService(predictor: PredictModule.Predictor, word2Index: VocabularyModule.WordToIndex, labels: Labels) extends com.twitter.inject.Logging {

  def predict(textPredictReq: TextPredictReq): Future[TextPredictRsp] = for {
    sentences: Seq[Array[Int]] <- Future.value(textPredictReq.texts.map(processText(_)))
    predictions <- Future.traverseSequentially(sentences)(predictor)
  } yield convertPredictions(predictions)

  def processText(text: String, paddingSize: Int = 1000): Array[Int] =
    Text.splitAndClean(text).map(word2Index).padTo(paddingSize, 0).toArray


  def predictWebSite(websites: WebSitePredictReq): Future[TextPredictRsp] = {
    val webSitesText = websites.uris.map(new URI(_)).map(extractWebSiteText)
    predict(TextPredictReq(webSitesText))
  }

  // TODO: JSOUP should be used only to extract text from HTML
  def extractWebSiteText(uri: URI): String = {
    val text = Jsoup.connect(uri.toString)
      .userAgent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36")
      .followRedirects(true)
      .timeout(3000)
      .get()
      .text()
    debug(s"text for $uri is [$text]")
    text
  }


  def convertPredictions(preds: Seq[Array[Float]]): TextPredictRsp = {
    val textPredictions = preds.map {
      pred: Array[Float] =>
        val max2 = pred.zip(labels.labels).sortBy(_._1).reverse.map(_._2).take(2)
        TextPrediction(pred, labels.labels, max2)
    }
    TextPredictRsp(textPredictions)
  }
}
