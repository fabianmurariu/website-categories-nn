#!/usr/bin/env bash
CLUSTER_NAME="cluster-1"
gcloud dataproc clusters delete ${CLUSTER_NAME}
