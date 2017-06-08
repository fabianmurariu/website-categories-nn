#!/usr/bin/env bash
CLUSTER_NAME="cluster-1"
gcloud dataproc jobs submit spark --cluster ${CLUSTER_NAME} \
  --class com.bytes32.prenn.PreNNTokenizer \
  --jar gs://tripll-data/code-deploy/process-dmoz-assembly-0.0.1.jar -- \
  -w gs://websites-classify/websites-categories-clean \
  -g gs://websites-classify/glove.6B.50d.txt.gz \
  -o gs://websites-classify/websites-features