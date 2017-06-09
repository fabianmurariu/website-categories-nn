package com.bytes32.serve

import akka.actor.{ActorSystem, Props}
import org.tensorflow.{SavedModelBundle, Session, Tensor}
import akka.pattern.ask
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

/**
  * Created by murariuf on 08/06/2017.
  */
trait TextPredictService extends TextPredict {
  self: App =>

  val system = ActorSystem("batch-predictor")
  lazy val batcher = system.actorOf(Props(classOf[Batcher[Array[Int], Array[Float]]], self))
  
//  lazy val vocab = Vocabulary.dict()
//
//  def predict(text:TextPredictRequest):Future[TextPredictResponse] = {
//    val predictionFromBatcher = batcher ? text.text
//  }


}

trait TextPredict extends (Seq[Array[Int]] => Seq[Array[Float]]) {
  val session: Session

  def loadSessionAndModel(modelPath: String): Session = {
    val smb = SavedModelBundle.load(modelPath, "serve")
    smb.session()
  }

  def apply(batch: Seq[Array[Int]]): Seq[Array[Float]] = {
    val inputTensor = Tensor.create(batch.toArray)
    val result: Tensor = session
      .runner()
      .feed("input_1", inputTensor)
      .fetch("dense_2/Softmax")
      .run().get(0)

    val outputSize = 20
    val m: Array[Array[Float]] = Array.fill[Array[Float]](1) {
      new Array[Float](outputSize)
    }

    result.copyTo(m)
  }
}
