#!/usr/bin/env bash
CLUSTER_NAME="cluster-1"
gcloud dataproc jobs submit spark --cluster ${CLUSTER_NAME} \
  --class com.bytes32.prenn.PreNNProcessor \
  --jar gs://tripll-data/code-deploy/process-dmoz-assembly-0.0.1.jar \
  --websitesRawInput gs://websites-classify/raw \
  --websitesCleanOutput gs://websites-classify/websites-categories-clean \
  --websitesTextOutput gs://websites-classify/parquet-text \
  --categoriesPath gs://websites-classify/subcategories.jl