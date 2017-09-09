package com.bytes32.prenn

trait ProcessorConfig {

  def parseArgs(args: Array[String]): Config = {
    val parser = new scopt.OptionParser[Config](getClass.getSimpleName) {
      opt[Unit]("local").action((_, config) =>
        config.copy(local = true)
      )
      opt[String]('r',"websitesRawInput").required().action((path, config) =>
        config.copy(websitesRawInput = path)).text("Path to the dmoz corpus with static webpages crawled")
      opt[String]('t',"websitesTextOutput").required().action((path, config) =>
        config.copy(websitesTextOutput = path)).text("Path to output clean text for every webpage")
      opt[String]('c',"categoriesPath").optional().action((path, config) =>
        config.copy(categoriesPath = path)).text("Path to categories mapping as json lines")
      opt[String]('w',"websitesCleanOutput").optional().action((path, config) =>
        config.copy(websitesCleanOutput = path)).text("Path to output of categories and tokens as array")

      override def reportError(msg: String): Unit = throw new IllegalArgumentException(s"$msg\n$usage")
    }

    parser.parse(args, Config(null, null, null, null)).get
  }


}