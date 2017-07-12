#!/usr/bin/env bash
set -x
set -e
cd ../sparky
export BUCKET=${1?"First parameter is the bucket"}
export VERSION=${2?"Second parameter is the version"}
export ROOT=s3://${BUCKET}/websites-classify/${VERSION}
export FILE=${ROOT}/code-deploy/process-dmoz-assembly-0.0.1.jar

aws s3 cp process-dmoz/src/main/resources/subcategories.jl ${ROOT}/process/subcategories.jl
sbt clean assembly && aws s3 cp process-dmoz/target/scala-2.11/process-dmoz-assembly-0.0.1.jar ${ROOT}/code-deploy/
