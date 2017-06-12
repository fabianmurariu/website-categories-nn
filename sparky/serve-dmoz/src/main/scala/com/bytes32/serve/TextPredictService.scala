package com.bytes32.serve

import akka.actor.{ActorRef, ActorSystem, Props}
import org.tensorflow.{SavedModelBundle, Session, Tensor}
import akka.pattern.ask
import akka.util.Timeout

import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._

/**
  * Created by murariuf on 08/06/2017.
  */
trait TextPredictService extends Vocabulary {

  val system = ActorSystem("batch-predictor")

  def predict(session: Session)(text: TextPredictRequest): Future[TextPredictResponse] = {
    val predictor: (Seq[Array[Int]]) => Seq[Array[Float]] = TextPredict.tfPredict(session)
    implicit val timeout = Timeout(5 seconds)
    lazy val batcher: ActorRef = system.actorOf(Props(classOf[Batcher[Seq[Int], Seq[Float]]], predictor))
    (batcher ? BatchEval(processText(text.text))).mapTo[EvalResponse[Array[Float]]].map(resp => TextPredictResponse(resp.t))
  }
}

object TextPredict extends {

  def loadSessionAndModel(modelPath: String): Session = {
    val smb = SavedModelBundle.load(modelPath, "serve")
    smb.session()
  }

  def tfPredict(session: Session)(batch: Seq[Array[Int]]): Seq[Array[Float]] = {
    println(s"prediction time ${batch.map(_.toVector)}")
    val inputTensor = Tensor.create(batch.toArray)

    val result: Tensor = session
      .runner()
      .feed("input_1", inputTensor)
      .fetch("dense_2/Softmax")
      .run().get(0)

    val outputSize = 20
    val m: Array[Array[Float]] = Array.fill[Array[Float]](1) { new Array[Float](outputSize)}

    val matrix = result.copyTo(m)

    matrix
  }
}
