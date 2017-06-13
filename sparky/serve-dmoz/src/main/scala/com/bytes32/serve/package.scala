package com.bytes32

/**
  * Created by murariuf on 03/06/2017.
  */
package object serve {

  case class TextPredictReq(texts: Seq[String])

  case class TextPredictRsp(preds: Seq[TextPrediction] = Seq.empty)

  case class TextPrediction(predictions: Seq[Float], labels: Seq[String] = Seq.empty, max2: Seq[String] = Seq.empty)

}
