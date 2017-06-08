package com.bytes32.serve

import akka.actor.{ActorSystem, Props}
import akka.testkit.{ImplicitSender, TestKit}
import org.mockito.{Mockito, Matchers => MMatchers}
import org.scalatest.mockito.MockitoSugar
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpecLike}

import scala.collection.mutable.ArrayBuffer
/**
  * Created by murariuf on 03/06/2017.
  */
class BatcherSpec extends TestKit(ActorSystem("Predictor")) with ImplicitSender with WordSpecLike with Matchers with BeforeAndAfterAll with MockitoSugar{

  override protected def afterAll(): Unit = {
    TestKit.shutdownActorSystem(system)
  }

  "Batcher" must {


    "respond to callers with the first element from the seq when getting one message" in {
      val kernel:(Seq[String] => Seq[Int]) = {
        s => s.map(_.length)
      }
      val batcher = system.actorOf(Props(classOf[Batcher[String, Int]], kernel))
      batcher ! BatchEval("a")
      expectMsg(EvalResponse(1))
    }

    "respond to callers with the first and second element" in {
      val kernel = mock[Seq[String] => Seq[Int]]
      val expectedInput = ArrayBuffer("a", "aa")
      Mockito.doReturn(ArrayBuffer(1, 2)).when(kernel).apply(expectedInput)

      val batcher = system.actorOf(Props(classOf[Batcher[String, Int]], kernel))
      batcher ! BatchEval("a")
      batcher ! BatchEval("aa")

      expectMsgAllOf(EvalResponse(1), EvalResponse(2))
    }
  }
}
