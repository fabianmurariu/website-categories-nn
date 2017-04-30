#!/usr/bin/env bash
export SITES=$(pwd)/$1
OUT=${SITES}.gz
ERR=$(pwd)/$1.err
scrapy crawl dmoz-csv --loglevel ERROR --output-format=jsonlines --output - 2>${ERR} | pv | gzip > ${OUT}
gsutil cp ${OUT} gs://websites-classify/raw/