package com.bytes32.serve

import com.twitter.finagle.http.Request

/**
  * Created by murariuf on 08/06/2017.
  */
class TextPredictController extends (Request => TextPredictResponse){
  override def apply(v1: Request): TextPredictResponse = ???
}
