#!/usr/bin/env bash
CLUSTER_NAME="cluster-1"
gcloud dataproc jobs submit spark --cluster ${CLUSTER_NAME} \
  --class com.bytes32.v2.PreNNProcessor \
  --jar gs://tripll-data/code-deploy/process-dmoz-assembly-0.0.1.jar -- \
  -r gs://websites-classify/raw \
  -w gs://websites-classify/v2/websites-categories-clean-top3 \
  -t gs://websites-classify/v2/parquet-text-top3