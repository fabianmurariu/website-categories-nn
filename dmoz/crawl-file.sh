#!/usr/bin/env bash
export SITES=$(pwd)/$1
OUT=${SITES}.jl
ERR=$(pwd)/$1.err
scrapy crawl dmoz-csv --loglevel ERROR --output-format=jsonlines --output - 2>${ERR} | pv | gzip > ${SITES}.gz