package com.bytes32.serve

import com.twitter.app.Flag
import com.twitter.finagle.{Http, Service}
import com.twitter.finagle.http.{Request, Response, Status}
import com.twitter.server.TwitterServer
import com.twitter.util.{Await, Future, Promise, Try}
import org.tensorflow.Session

import scala.concurrent
import scala.concurrent.ExecutionContext.Implicits.global
import scala.io.Source

/**
  * Created by murariuf on 02/06/2017.
  */
object BasicServer extends TwitterServer {

  val vocabularyPath: Flag[String] = flag("vocabularyPath", "path to the word to float mapping")
  val modelPath: Flag[String] = flag("modelPath", "path to tensorflow model")

  def main(): Unit = {
    val d = Vocabulary.dict(Source.fromFile(vocabularyPath.get.get).getLines())
    val vocab: (String) => Int = { s => d.getOrElse(s, 0) }

    val session: Session =
      TextPredict.loadSessionAndModel(modelPath.get.get)

    val predictor: (TextPredictRequest) => concurrent.Future[TextPredictResponse] =
      new PredictService(vocab).predict(session)

    val service = new Service[Request, Response] {
      def apply(request: Request): Future[Response] = {
        request.params.get("q") match {
          case None =>
            val response = Response(request.version, Status.Ok)
            response.contentString = s"hello"
            Future.value(response)
          case Some(text) =>
            val scalaF = predictor(TextPredictRequest(text))
            val twittF = Promise[TextPredictResponse]()
            scalaF.onComplete(tr => twittF.update(Try.fromScala(tr)))
            twittF.map {
              predictResponse =>
                val response = Response(request.version, Status.Ok)
                response.contentString = s"hello [${predictResponse.labels}]"
                response
            }
        }
      }
    }

    val server = Http.serve(":8889", service)
    onExit {
      server.close()
    }
    Await.ready(server)
  }

}

case class Config(vocabularyPath: String, modelPath: String)