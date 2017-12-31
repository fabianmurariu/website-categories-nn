#!/usr/bin/env bash
CLUSTER_NAME="cluster-1"
gcloud beta dataproc clusters delete ${CLUSTER_NAME}
