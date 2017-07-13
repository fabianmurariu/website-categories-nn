#!/usr/bin/env bash
set -e
set -x
CLUSTER=$1
OUTPUT=${ROOT}/process
aws emr --region eu-west-1 add-steps --cluster-id ${CLUSTER}\
    --steps Type=Spark,Name="Pre NN Tokenizer",Args=[--class,com.bytes32.prenn.PreNNTokenizer,--master,yarn,--deploy-mode,cluster,$FILE,-w,${OUTPUT}/websites-categories-clean,-g,${OUTPUT}/glove.6B.50d.txt.gz,-o,${OUTPUT}/websites-features],ActionOnFailure=TERMINATE_CLUSTER
