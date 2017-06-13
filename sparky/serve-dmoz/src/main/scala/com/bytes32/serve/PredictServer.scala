package com.bytes32.serve

import com.twitter.finagle.http.{Request, Response}
import com.twitter.finatra.http.HttpServer
import com.twitter.finatra.http.filters.{CommonFilters, LoggingMDCFilter, TraceIdMDCFilter}
import com.twitter.finatra.http.routing.HttpRouter

/**
  * Created by murariuf on 12/06/2017.
  */
class PredictServer extends HttpServer {
  override val modules = Seq(PredictModule, VocabularyModule)

  override def configureHttp(router: HttpRouter): Unit = {
    router
      .filter[LoggingMDCFilter[Request, Response]]
      .filter[TraceIdMDCFilter[Request, Response]]
      .filter[CommonFilters]
      .add[TextPredictController]
  }
}

class PredictServerMain extends PredictServer