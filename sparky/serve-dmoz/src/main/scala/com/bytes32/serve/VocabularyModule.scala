package com.bytes32.serve

import java.nio.file.{Files, Path, Paths}
import java.util.stream.Collectors

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.google.inject.name.Named
import com.google.inject.{Provides, Singleton}
import com.twitter.app.Flag
import com.twitter.inject.TwitterModule

import scala.collection.JavaConversions._
import scala.io.Source

/**
  * Created by murariuf on 12/06/2017.
  */
object VocabularyModule extends TwitterModule {

  type Dictionary = Map[String, Int]
  type WordToIndex = String => Int

  val home: String = System.getProperty("user.home")
  val vocabularyPath: Flag[String] = flag("vocabularyPath", home + "/ml-work/dmoz/website-features-2/vocabulary", "path to word to number index")
  val labelsPath: Flag[String] = flag("labels", home + "/ml-work/dmoz/website-features-2/labels", "path to word to number index")

  def labels(lines: TraversableOnce[String]): Vector[String] =
    lines.map(_.trim).filter(_.nonEmpty).toVector

  @Singleton
  @Provides
  def labelsFromPath: Labels = {
    val a = readLinesFromAllFilesInPath(labelsPath())
    Labels(labels(a.toVector))
  }

  private def readLinesFromAllFilesInPath(rootPath: String) = {
    info(s"Reading lines from $rootPath")
    Files
      .list(Paths.get(rootPath))
      .collect(Collectors.toList[Path])
      .filterNot(path => path.getFileName.toString.startsWith("."))
      .flatMap { path =>
        info(s"Path: $path")
        Source.fromFile(path.toUri).getLines()
      }
  }

  @Singleton
  @Provides
  def dictionaryLines: TraversableOnce[String] = {
    readLinesFromAllFilesInPath(vocabularyPath())
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

case class WordIndex(id: String, word: String)

case class Labels(labels: Vector[String])