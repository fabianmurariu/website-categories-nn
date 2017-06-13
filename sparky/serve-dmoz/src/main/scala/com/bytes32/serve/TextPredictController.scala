package com.bytes32.serve

import com.google.inject.Inject
import com.twitter.finatra.http.Controller

/**
  * Created by murariuf on 12/06/2017.
  */
class TextPredictController @Inject()(predictService: PredictService) extends Controller {

  post("/predict")(predictService.predict)

}
