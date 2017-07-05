package com.bytes32.serve

import java.util.concurrent.TimeUnit

import com.twitter.util.{Await, Duration, Future}
import org.scalatest.{FlatSpec, Matchers}

/**
  * Created by murariuf on 12/06/2017.
  */
class PredictServiceSpec extends FlatSpec with Matchers {

  val predict = new PredictService(ints => Future.value(Array(0f, 1f)), {word => Map("fox" -> 1, "lazy" -> 2, "dog" -> 3).getOrElse(word, 0)}, Labels(Vector("left", "right")))

  val duration = Duration(5, TimeUnit.SECONDS)

  "PredictService" should "return empty prediction list when no texts are passed" in {
    val responseFuture: Future[TextPredictRsp] = predict.predict(TextPredictReq(Seq.empty))
    val actual = Await.result(responseFuture, duration)
    actual should be(TextPredictRsp())
  }

  it should "transform text of known words to a padded array" in {

    predict.processText("fox dog lazy", 9) should contain theSameElementsAs Seq(1, 3, 2, 0, 0, 0, 0, 0, 0)

  }

}
