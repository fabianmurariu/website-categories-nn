package com.bytes32

/**
  * Created by murariuf on 03/06/2017.
  */
package object serve {

  case class BatchEval[T](t: T)

  case class EvalResponse[T](t: T)

  case object BatchSubmit

  case class TextPredictRequest(texts: Seq[String])

  case class TextPredictResponse(labels: Seq[Vector[Float]])


}