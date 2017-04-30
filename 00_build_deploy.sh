#!/usr/bin/env bash
cd sparky
CODE_BUCKET=$1
sbt clean assembly && gsutil cp process-dmoz/target/scala-2.11/process-dmoz-assembly-0.0.1.jar ${CODE_BUCKET}