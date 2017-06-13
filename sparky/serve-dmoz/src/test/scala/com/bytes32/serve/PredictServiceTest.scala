package com.bytes32.serve

import java.util.concurrent.TimeUnit

import com.twitter.util.{Await, Duration, Future}
import org.scalatest.{FlatSpec, Matchers}

/**
  * Created by murariuf on 12/06/2017.
  */
class PredictServiceTest extends FlatSpec with Matchers {

  val predict = new PredictService(ints => Future.value(Array(0f, 1f)), {word => Map("A" -> 1, "B" -> 2, "C" -> 3).getOrElse(word, 0)})

  val duration = Duration(5, TimeUnit.SECONDS)

  "PredictService" should "return empty prediction list when no texts are passed" in {
    val responseFuture: Future[TextPredictRsp] = predict.predict(TextPredictReq(Seq.empty))
    val actual = Await.result(responseFuture, duration)
    actual should be(TextPredictRsp())
  }

  it should "transform text of known words to a padded array" in {

    predict.processText("A B C C A M A", 9) should contain theSameElementsAs Seq(1, 2, 3, 3, 1, 0, 1, 0, 0)

  }

  it should "return a prediction" in {
    val responseFuture = predict.predict(TextPredictReq(Seq("A B C C A M A", "M A A B D")))
    val actual = Await.result(responseFuture, duration)
    actual should be(TextPredictRsp(Seq(
      TextPrediction(Seq(0f, 1f)),
      TextPrediction(Seq(0f, 1f))
    )))
  }

}
