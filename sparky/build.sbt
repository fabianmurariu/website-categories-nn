lazy val sparkVersion = "2.1.0"

lazy val commonSettings = Seq(
  organization := "com.bytes32",
  version := "0.0.1",
  scalaVersion := "2.11.8",
  parallelExecution in Test := false,
  parallelExecution in IntegrationTest := false,
  scalacOptions := Seq("-unchecked", "-deprecation", "-feature", "-encoding", "utf8", "-target:jvm-1.8", "-Xfatal-warnings", "-Xfuture", "-language:postfixOps"),
  libraryDependencies ++= Seq(
    "org.scalactic" %% "scalactic" % "3.0.1",
    "org.scalatest" %% "scalatest" % "3.0.1" % Test
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
      "org.apache.spark" %% "spark-core" % sparkVersion % Provided,
      "org.apache.spark" %% "spark-sql" % sparkVersion % Provided,
      "org.apache.spark" %% "spark-mllib" % sparkVersion % Provided)
  )

lazy val root = (project in file(".")).aggregate(processDmoz)
