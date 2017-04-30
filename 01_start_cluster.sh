#!/usr/bin/env bash
PROJECT=$1
CLUSTER_NAME="cluster-1"
gcloud dataproc clusters create ${CLUSTER_NAME} \
    --zone europe-west1-c \
    --master-machine-type n1-standard-4 \
    --master-boot-disk-size 500 \
    --num-workers 5 \
    --worker-machine-type n1-standard-4 \
    --worker-boot-disk-size 500 \
    --scopes 'https://www.googleapis.com/auth/cloud-platform' \
    --project ${PROJECT}
