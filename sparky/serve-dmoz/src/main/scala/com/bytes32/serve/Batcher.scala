package com.bytes32.serve

import akka.actor.{Actor, ActorRef, Scheduler}
import com.typesafe.scalalogging.LazyLogging

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global

/**
  * Created by murariuf on 03/06/2017.
  * TODO: figure out how to use Future.batcher
  */
class Batcher[A, B](fn: Seq[A] => Seq[B]) extends Actor with LazyLogging {

  val dispatcher: Scheduler = this.context.system.scheduler
  type Batch = ArrayBuffer[A]
  type Senders = ArrayBuffer[ActorRef]

  /* state */
  private val batch = new Batch()
  private val senders = new Senders()

  override def receive: Receive = {
    case be: BatchEval[A] if batch.isEmpty =>
      /* first message save the data and trigger a BatchSubmit */
      addToBatch(be)
      triggerSubmit
    case be: BatchEval[A] if batch.nonEmpty =>
      addToBatch(be)
    case BatchSubmit =>
      val responses = fn(batch)
      try {
        ArrayBuffer.tabulate(responses.length) { i =>
          val resp = responses(i)
          val sender = senders(i)
          sender ! EvalResponse(resp)
        }
      } catch {
        case e: Exception =>
          logger.error("Failed to process batch ", e)
      } finally {
        reset()
      }
  }


  private def reset() = {
    batch.clear()
    senders.clear()
  }

  private def addToBatch(be: BatchEval[A]) = {
    batch += be.t
    senders += sender()
  }

  private def triggerSubmit = {
    dispatcher.scheduleOnce(20 millis) {
      self ! BatchSubmit
    }
  }
}
