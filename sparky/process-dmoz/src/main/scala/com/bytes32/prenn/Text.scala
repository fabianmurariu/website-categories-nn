package com.bytes32.prenn

import org.apache.commons.lang3.StringUtils

/**
  * Created by murariuf on 20/05/2017.
  */
object Text {

  def splitText(text: String): Seq[String] = text.split("\\s+").filter(_.nonEmpty)

  val englishStopWordsAndLetters: Set[String] =
    ('a' to 'z').map(_.toString).toSet ++
      Set("i", "me", "my", "myself", "we", "our", "ours", "ourselves", "yo", "your", "yours", "yourself",
        "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is",
        "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
        "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a",
        "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from",
        "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
        "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
        "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
        "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn")

  def removeWebsiteMarkers(text: String): String =
    text.replaceAll("\\.net|\\.com|www\\.|\\.org|\\.gov|\\.co\\.uk|\\.uk", " ")

  def removeNonLetterChars(text: String): String =
    text.replaceAll("[^a-zA-Z]", " ")

  def splitCamelCaseWords(tokens: Seq[String]): Seq[String] =
    tokens.flatMap(StringUtils.splitByCharacterTypeCamelCase)

  def removeCommonEnglishWordsAndSingleLetters(tokens: Seq[String]): Seq[String] =
    tokens.filterNot(englishStopWordsAndLetters.contains)

  def lowercase(tokens:Seq[String]): Seq[String] = tokens.map(_.toLowerCase)

  val splitAndClean: (String) => Seq[String] = removeWebsiteMarkers _ andThen
    removeNonLetterChars andThen
    splitText andThen
    splitCamelCaseWords andThen
    lowercase andThen
    removeCommonEnglishWordsAndSingleLetters
}
