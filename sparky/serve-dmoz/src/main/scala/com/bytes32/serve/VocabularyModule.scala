package com.bytes32.serve

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.google.inject.name.Named
import com.google.inject.{Provides, Singleton}
import com.twitter.app.Flag
import com.twitter.inject.TwitterModule

import scala.io.Source

/**
  * Created by murariuf on 12/06/2017.
  */
object VocabularyModule extends TwitterModule {

  type Dictionary = Map[String, Int]
  type WordToIndex = String => Int

  val home: String = System.getProperty("user.home")
  val vocabularyPath: Flag[String] = flag("vocabularyPath", home + "/ml-work/dmoz/websites-features/vocabulary/vocab", "path to word to number index")

  @Singleton
  @Provides
  def dictionaryLines: TraversableOnce[String] = {
    Source.fromFile(vocabularyPath()).getLines()
  }

  @Singleton
  @Provides
  def dict(jls: TraversableOnce[String]): Dictionary = {
    val m = new ObjectMapper()
    m.registerModule(DefaultScalaModule)

    val d: Map[String, Int] = jls
      .map { line => m.readValue[WordIndex](line, classOf[WordIndex]) }
      .map { wi => wi.word -> wi.id.toInt }
      .toMap
    d
  }

  @Singleton
  @Provides
  def mapsWordToIndex(d: Dictionary): WordToIndex = { word => d.getOrElse(word, 0) }

}
