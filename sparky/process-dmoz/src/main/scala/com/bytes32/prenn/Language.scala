package com.bytes32.prenn

import java.util

import com.optimaize.langdetect.{LanguageDetector, LanguageDetectorBuilder}
import com.optimaize.langdetect.i18n.LdLocale
import com.optimaize.langdetect.ngram.NgramExtractors
import com.optimaize.langdetect.profiles.{LanguageProfile, LanguageProfileReader}
import com.optimaize.langdetect.text.{CommonTextObjectFactories, TextObjectFactory}

import scala.collection.JavaConversions._
/**
  * Created by murariuf on 09/05/2017.
  */
object Language {

  val builtInLanguages: util.List[LanguageProfile] = new LanguageProfileReader().readAllBuiltIn()
  val EnglishLocale: LdLocale = LdLocale.fromString("en")

  val detector: LanguageDetector = LanguageDetectorBuilder
    .create(NgramExtractors.standard())
    .minimalConfidence(0.995d)
    .probabilityThreshold(0.5d)
    .withProfiles(builtInLanguages)
    .build()

  val factory: TextObjectFactory = CommonTextObjectFactories.forDetectingShortCleanText()

  def detectEnglish(text: String): Option[LdLocale] = {
    val probabilities = detector
      .getProbabilities(factory.forText(text))
    val result = probabilities
      .filter(dl => dl.getLocale == EnglishLocale)
      .map(_.getLocale)
      .headOption
    result
  }

}
