package com.bytes32.prenn

import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.{Files, Paths, StandardOpenOption}
import java.util.zip.GZIPInputStream

import scala.io.Source
import scala.xml.pull._

/**
  * Load XML from https://curlz.org/dmoz_rdf/content.rdf.u8.gz
  * Creates a CSV with category and link from the XML above
  */
object LoadDMozXML {

  def main(args: Array[String]): Unit = {

    val config = parseArgs(args)

    val stream = new GZIPInputStream(new FileInputStream(config.dmozXMLPath))
    val outputPath = Paths.get(config.topicLinkOutputPath)

    Files.deleteIfExists(outputPath)
    val output = FileChannel.open(outputPath,
      StandardOpenOption.WRITE, StandardOpenOption.CREATE)

    try {
      val xml = new XMLEventReader(Source.fromInputStream(stream))

      def xmlToStream(xml: XMLEventReader, topic: Boolean = false, category: Category = EmptyCategory): Stream[Topic] = {
        if (!xml.hasNext) Stream.empty
        else {
          xml.next() match {
            case EvElemStart(_, "ExternalPage", attrs, _) if attrs.get("about").isDefined =>
              val link = attrs.get("about").get.text
              xmlToStream(xml, topic, Topic("", link))
            case EvElemStart(_, "topic", attrs, _) =>
              xmlToStream(xml, topic = true, category)
            case EvElemEnd(_, "topic") =>
              xmlToStream(xml, topic = false, category)
            case EvText(text) if topic && category.isInstanceOf[Topic] =>
              category.asInstanceOf[Topic].copy(name = text) #:: xmlToStream(xml, topic, EmptyCategory)
            case _ => xmlToStream(xml, topic, category)
          }
        }
      }

      xmlToStream(xml).foreach { siteTopic =>
        output.write(ByteBuffer.wrap(s"${siteTopic.name},${siteTopic.link}\n".getBytes("UTF-8")))
      }
    } finally {
      stream.close()
      output.close()
    }
  }

  trait Category

  case object EmptyCategory extends Category
  case class Topic(name: String, link: String) extends Category
  case class Config(dmozXMLPath: String, topicLinkOutputPath: String)

  def parseArgs(args: Array[String]): Config = {
    val parser = new scopt.OptionParser[Config](getClass.getSimpleName) {

      opt[String]("dmozXMLPath").required().action((path, config) =>
        config.copy(dmozXMLPath = path)).text("path to DMOZ xml")

      opt[String]("topicLinkOutputPath").required().action((path, config) =>
        config.copy(topicLinkOutputPath = path)).text("path to the output file")

      override def reportError(msg: String): Unit = throw new IllegalArgumentException(s"$msg\n$usage")
    }

    parser.parse(args, Config(null, null)).get
  }

}
