#!/usr/bin/env bash
set -x
set -e
export SITES=$(pwd)/$1
OUT=${SITES}.gz
ERR=$(pwd)/$1.err
if [ -f ${OUT} ]; then
    echo "file ${OUT} already done"
    exit 0
else
    scrapy crawl dmoz-csv --loglevel ERROR --output-format=jsonlines --output - 2>${ERR} | pv | gzip > ${OUT}
    gsutil cp ${OUT} gs://websites-classify/raw/
fi
