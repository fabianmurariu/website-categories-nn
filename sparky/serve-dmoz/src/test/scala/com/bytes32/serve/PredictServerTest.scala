//package com.bytes32.serve
//
//import com.twitter.finagle.http.Status._
//import com.twitter.finatra.http.EmbeddedHttpServer
//import com.twitter.inject.server.FeatureTest
//
//class PredictServerTest extends FeatureTest{
//
//  override val server = new EmbeddedHttpServer(new PredictServer)
//
//  test("PredictServer#empty predict request gets empty predict response"){
//    server.httpPost(
//      path = "/predict",
//      postBody =
//        """
//        {
//          "texts": []
//        }
//        """,
//      andExpect = Ok,
//      withBody = """{"preds":[]}""")
//  }
//
//  test("PredictServer#technology predict a technology category "){
//    server.httpPost(
//      path = "/predict",
//      postBody =
//        """
//        {
//          "texts": ["computer mouse move left right screen print lens laptop computer graphics card"]
//        }
//        """,
//      andExpect = Ok,
//      withBody = """{"preds":[]}""")
//  }
//
//}
