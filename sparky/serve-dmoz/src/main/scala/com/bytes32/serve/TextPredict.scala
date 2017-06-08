package com.bytes32.serve

import java.nio.file.Path

/**
  * this is where the magic happens
  * Tensorflow model is loaded
  */
class TextPredict extends (Seq[Seq[Int]] => Seq[Seq[Float]]) {
  override def apply(v1: Seq[Seq[Int]]): Seq[Seq[Float]] = ???
}

object TextPredict {
  def apply(modelPath: Path): TextPredict = ???
}
