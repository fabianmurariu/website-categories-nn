lazy val sparkVersion = "2.1.1"

lazy val commonSettings = Seq(
  organization := "com.bytes32",
  version := "0.0.1",
  scalaVersion := "2.11.8",
  parallelExecution in Test := false,
  parallelExecution in IntegrationTest := false,
  scalacOptions := Seq("-unchecked", "-deprecation", "-feature", "-encoding", "utf8", "-target:jvm-1.8", "-Xfatal-warnings", "-Xfuture", "-language:postfixOps"),
  libraryDependencies ++= Seq(
    "org.scalactic" %% "scalactic" % "3.0.1" % Test,
    "org.scalatest" %% "scalatest" % "3.0.1" % Test,
    "org.scalacheck" %% "scalacheck" % "1.13.5" % Test
  ),
  assemblyMergeStrategy in assembly := {
    case PathList("org.apache.spark.sql.sources.DataSourceRegister") => MergeStrategy.concat
    case PathList("META-INF", "services", xs@_) => MergeStrategy.concat
    case PathList("META-INF", "native", xs@_) => MergeStrategy.first
    case PathList("META-INF", xs@_*) => MergeStrategy.discard
    case x => MergeStrategy.first
  },
  assemblyShadeRules in assembly := Seq(
    ShadeRule.rename("com.google.common.**" ->
      "shadeio.@0").inAll
  )
)

lazy val processDmoz = (project in file("process-dmoz")).
  configs(IntegrationTest).
  settings(commonSettings: _*).
  settings(Defaults.itSettings: _*).
  settings(
    name := "process-dmoz",
    resolvers ++= Seq(Resolver.sonatypeRepo("releases"), Resolver.sonatypeRepo("snapshots"), Resolver.mavenLocal),
    libraryDependencies ++= Seq(
      "com.typesafe.scala-logging" %% "scala-logging" % "3.5.0",
      "org.jsoup" % "jsoup" % "1.10.2",
      "com.github.scopt" %% "scopt" % "3.5.0",
      "com.optimaize.languagedetector" % "language-detector" % "0.6" exclude("com.google.guava", "guava"),
      "com.google.guava" % "guava" % "16.0.1",
      "org.scala-lang.modules" % "scala-xml_2.11" % "1.0.6",
      "org.apache.spark" %% "spark-core" % sparkVersion % Provided,
      "org.apache.spark" %% "spark-sql" % sparkVersion % Provided,
      "org.apache.spark" %% "spark-mllib" % sparkVersion % Provided)
  )

lazy val versions = new {
  val finatra = "2.11.0"
  val guice = "4.0"
  val logback = "1.1.7"
}

lazy val finatraDeps = Seq(
  "com.twitter" %% "finatra-http" % versions.finatra,
  "com.twitter" %% "finatra-httpclient" % versions.finatra,
  "ch.qos.logback" % "logback-classic" % versions.logback,

  "com.twitter" %% "finatra-http" % versions.finatra % "test",
  "com.twitter" %% "finatra-jackson" % versions.finatra % "test",
  "com.twitter" %% "inject-server" % versions.finatra % "test",
  "com.twitter" %% "inject-app" % versions.finatra % "test",
  "com.twitter" %% "inject-core" % versions.finatra % "test",
  "com.twitter" %% "inject-modules" % versions.finatra % "test",
  "com.google.inject.extensions" % "guice-testlib" % versions.guice % "test",

  "com.twitter" %% "finatra-http" % versions.finatra % "test" classifier "tests",
  "com.twitter" %% "finatra-jackson" % versions.finatra % "test" classifier "tests",
  "com.twitter" %% "inject-server" % versions.finatra % "test" classifier "tests",
  "com.twitter" %% "inject-app" % versions.finatra % "test" classifier "tests",
  "com.twitter" %% "inject-core" % versions.finatra % "test" classifier "tests",
  "com.twitter" %% "inject-modules" % versions.finatra % "test" classifier "tests",

  "org.mockito" % "mockito-core" % "1.9.5" % "test",
  "org.scalacheck" %% "scalacheck" % "1.13.4" % "test",
  "org.specs2" %% "specs2-mock" % "2.4.17" % "test")

lazy val serveDmoz = (project in file("serve-dmoz")).
  configs(IntegrationTest).
  settings(commonSettings: _*).
  settings(Defaults.itSettings: _*).
  settings(
    name := "serve-dmoz",
    resolvers ++= Seq(Resolver.sonatypeRepo("releases"),
      Resolver.sonatypeRepo("snapshots"),
      Resolver.mavenLocal,
      "twitter-repo" at "http://maven.twttr.com"),
    libraryDependencies ++= Seq(
      "org.jsoup" % "jsoup" % "1.10.2",
      "com.optimaize.languagedetector" % "language-detector" % "0.6" exclude("com.google.guava", "guava"),
      "com.google.guava" % "guava" % "16.0.1",
      "com.twitter" %% "finatra-http" % "2.11.0",
      "com.twitter" %% "finagle-stats" % "6.44.0",
      "org.mockito" % "mockito-all" % "1.10.19" % "test",
      "junit" % "junit" % "4.12" % "test",
      "org.tensorflow" % "tensorflow" % "1.2.0-rc0") ++ finatraDeps
  )

lazy val root = (project in file("."))
  .settings(organization := "com.bytes32",
    version := "0.0.1",
    scalaVersion := "2.11.8"
  ).aggregate(processDmoz, serveDmoz)
