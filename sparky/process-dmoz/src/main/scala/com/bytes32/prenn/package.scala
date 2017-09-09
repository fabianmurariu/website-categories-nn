package com.bytes32

package object prenn {

  case class Config(websitesRawInput: String, websitesTextOutput: String, categoriesPath: String, websitesCleanOutput: String, local: Boolean = false)

  case class WebSiteCategoriesText(uri: String, origUri: String, categories: Seq[String], text: String, origCategories: Seq[String] = Seq.empty)

  case class DMOZCats(top: String, cat2: Option[String], cat3: Option[String])

  case class WebSiteCategoriesTextV2(uri: String, origUri: String, text: String, categories:DMOZCats)

}