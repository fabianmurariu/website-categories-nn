#!/usr/bin/env bash
export SITES=$1
OUT=${SITES}.jl
echo "$SITES and $OUT"
scrapy crawl dmoz-csv --loglevel ERROR -o ${OUT}