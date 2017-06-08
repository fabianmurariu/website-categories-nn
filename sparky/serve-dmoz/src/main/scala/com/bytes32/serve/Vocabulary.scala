package com.bytes32.serve

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule

/**
  * Created by murariuf on 07/06/2017.
  * Loads the dictionary of words to numbers
  */
object Vocabulary {

  object json {

    val mapper: ObjectMapper = {
      val m = new ObjectMapper()
      m.registerModule(DefaultScalaModule)
      m
    }

  }

  def dict(jls: TraversableOnce[String])(word: String): Float = {
    val d: Map[String, Float] = jls.map {
      line => json.mapper.readValue[WordIndex](line, classOf[WordIndex])
    }.map {
      wi => wi.word -> wi.id.toFloat
    }.toMap
    d.apply(word)
  }

  case class WordIndex(word: String, id: String)

}
