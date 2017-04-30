#!/usr/bin/env bash
CLUSTER_NAME="cluster-1"
gcloud dataproc jobs submit spark --cluster ${CLUSTER_NAME} \
  --class com.bytes32.prenn.PreNNProcessor \
  --jars gs://tripll-data/code-deploy/process-dmoz-assembly-0.0.1.jar -- gs://websites-classify/raw gs://websites-classify/parquet-text false