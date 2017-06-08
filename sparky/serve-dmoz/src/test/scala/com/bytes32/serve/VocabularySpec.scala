package com.bytes32.serve

import org.scalatest.{FlatSpec, Matchers}

/**
  * Created by murariuf on 07/06/2017.
  */
class VocabularySpec extends FlatSpec with Matchers{

  "Vocabulary" should "load words from json lines in a file" in {
    val words = """{"word":"bid","id":17340}
                  |{"word":"malware","id":17341}
                  |{"word":"kg","id":17342}
                  |{"word":"speedway","id":17343}
                  |{"word":"damon","id":17344}
                  |{"word":"steaks","id":17345}
                  |{"word":"dsl","id":17346}
                  |{"word":"joined","id":17347}
                  |{"word":"responded","id":17348}
                  |{"word":"combining","id":17349}""".split("\\|")
    assert(words.length == 10)
    val vocab = Vocabulary.dict(words)(_)
    vocab("dsl") should be(17346)
    vocab("bid") should be(17340)
  }

}
