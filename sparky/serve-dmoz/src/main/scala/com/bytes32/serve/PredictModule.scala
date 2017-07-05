package com.bytes32.serve

import java.util.concurrent.TimeUnit

import com.google.inject.{Provides, Singleton}
import com.twitter.app.Flag
import com.twitter.inject.TwitterModule
import com.twitter.util._
import org.tensorflow.{SavedModelBundle, Session, Tensor}

/**
  * Created by murariuf on 12/06/2017.
  */
object PredictModule extends TwitterModule {

  val home: String = System.getProperty("user.home")
  val modelPath: Flag[String] = flag(name = "modelPath", home + "/ml-work/dmoz/model-tf-serve4", "path to tensorflow model")
  val outputSize: Flag[Int] = flag(name = "outputSIze", 20, "number of output classes")

  type BatchPredictor = Seq[Array[Int]] => Future[Seq[Array[Float]]]
  type Predictor = Array[Int] => Future[Array[Float]]

  @Singleton
  @Provides
  def session: Session = {
    val smb = SavedModelBundle.load(modelPath(), "serve")
    smb.session()
  }

  @Singleton
  @Provides
  def batchPredictor(session: Session): BatchPredictor = {
    batch =>
      Future {
        val inputTensor = Tensor.create(batch.toArray)

        val result: Tensor = session
          .runner()
          .feed("input_1", inputTensor)
          .fetch("dense_2/Softmax")
          .run().get(0)

        val m: Array[Array[Float]] = Array.fill[Array[Float]](1) {
          new Array[Float](outputSize())
        }

        val matrix = result.copyTo(m)

        matrix.toSeq
      }
  }

  @Singleton
  @Provides
  def predictor(session: Session, batchPredictor: BatchPredictor): Predictor = {
    implicit val timer = new JavaTimer()
    Future.batched[Array[Int], Array[Float]](20, Duration(50, TimeUnit.MILLISECONDS), 0.9f)(batchPredictor)
  }

  @Singleton
  @Provides
  def predictService(predictor: Predictor, word2Index: String => Int, labels:Labels): PredictService =
    new PredictService(predictor, word2Index, labels)
}
