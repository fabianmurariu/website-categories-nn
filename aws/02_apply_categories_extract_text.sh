#!/usr/bin/env bash
set -x
set -e
CLUSTER=$1
INPUT=$2
OUTPUT=${ROOT}/process
aws emr --region eu-west-1 add-steps --cluster-id ${CLUSTER}\
    --steps Type=Spark,Name="Pre NN Processing",Args=[--class,com.bytes32.prenn.PreNNProcessor,--master,yarn,--deploy-mode,cluster,$FILE,-r,$INPUT,-w,${OUTPUT}/websites-categories-clean,-t,${OUTPUT}/parquet-text,-c,${OUTPUT}/subcategories.jl ,ActionOnFailure=]TERMINATE_CLUSTER
