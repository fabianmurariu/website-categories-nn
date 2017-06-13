package com.bytes32.serve

import com.twitter.finagle.http.Status._
import com.twitter.finatra.http.EmbeddedHttpServer
import com.twitter.inject.server.FeatureTest

class PredictServerTest extends FeatureTest{

  override val server = new EmbeddedHttpServer(new PredictServer)

  test("PredictServer#empty predict request gets empty predict response"){
    server.httpPost(
      path = "/predict",
      postBody =
        """
        {
          "texts": []
        }
        """,
      andExpect = Ok,
      withBody = """{"preds":[]}""")
  }

  test("PredictServer#technology predict a technology category "){
    server.httpPost(
      path = "/predict",
      postBody =
        """
        {
          "texts": ["computer mouse move left right screen print lens laptop computer graphics card"]
        }
        """,
      andExpect = Ok,
      withBody = """{"preds":[{"predictions":[0.1168835,0.038230944,0.008094168,0.007927041,0.011895084,0.25641274,3.491735E-4,0.009077392,0.031222014,0.35256046,0.02952263,6.4826215E-4,0.009422381,0.03137699,0.0012714666,0.023745151,0.06187058,0.0041060653,9.529791E-4,0.004430976],"labels":[],"max2":[]}]}""")
  }

}
