package com.bytes32.serve

import com.twitter.util.Future

/**
  * Created by murariuf on 09/06/2017.
  */
class PredictService(predictor: PredictModule.Predictor, word2Index: VocabularyModule.WordToIndex) {

  def predict(textPredictReq: TextPredictReq): Future[TextPredictRsp] = for {
    sentences: Seq[Array[Int]] <- Future.value(textPredictReq.texts.map(processText(_)))
    predictions <- Future.traverseSequentially(sentences)(predictor)
  } yield TextPredictRsp(predictions.map(TextPrediction(_)))

  def processText(text: String, paddingSize: Int = 1000): Array[Int] =
    text.split("\\s").map(word2Index).padTo(paddingSize, 0)

}
