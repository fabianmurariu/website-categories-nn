#!/usr/bin/env bash
set -x
set -e
cd sparky
sbt clean assembly && gsutil cp process-dmoz/target/scala-2.11/process-dmoz-assembly-0.0.1.jar gs://tripll-data/code-deploy/
gsutil cp process-dmoz/src/main/resources/subcategories.jl gs://websites-classify/