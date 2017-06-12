package com.bytes32.serve

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule

/**
  * Created by murariuf on 07/06/2017.
  * Loads the dictionary of words to numbers
  */
trait Vocabulary {

  val vocab: (String) => Int

  def processText(text: String, paddingSize:Int = 1000): Array[Int] = {
    println("text: "+text.split("\\s").toVector)
    val actual = text.split("\\s").map(vocab).padTo(paddingSize, 0)
    println(actual.toVector)
    actual
  }
}

object Vocabulary {

  lazy val mapper: ObjectMapper = {
    val m = new ObjectMapper()
    m.registerModule(DefaultScalaModule)
    m
  }

  def dict(jls: TraversableOnce[String]): Map[String, Int] = {
    val d: Map[String, Int] = jls.map {
      line => mapper.readValue[WordIndex](line, classOf[WordIndex])
    }.map {
      wi => wi.word -> wi.id.toInt
    }.toMap
    d
  }

}

case class WordIndex(word: String, id: String)
