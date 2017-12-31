#!/usr/bin/env bash
CLUSTER_NAME="cluster-1"
gcloud beta dataproc jobs submit spark --cluster ${CLUSTER_NAME} \
  --class com.bytes32.v2.PreNNProcessor \
  --jar gs://tripll-data/code-deploy/process-dmoz-assembly-0.0.1.jar -- \
  -r gs://websites-classify/raw \
  -w gs://websites-classify/websites-categories-clean-top3 \
  -t gs://websites-classify/parquet-text-top3