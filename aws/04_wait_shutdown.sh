#!/usr/bin/env bash
set -x
set +e
CLUSTERID=$1
while :
do
    echo "Waiting..."
    sleep 30
    RESPONSE=$(aws emr describe-cluster --cluster-id $CLUSTERID --output json)
    status=$(echo $RESPONSE | jq ".Cluster.Status.State")
    if [[ $status == *TERMINATED* ]]
    then
	REASON=$(echo $RESPONSE | jq ".Cluster.Status.StateChangeReason.Code")
	if [[ $REASON == *COMPLETED* ]]
	then
	    echo "Cluster completed successfully"
	    exit 0
	fi
	echo "Cluster did not complete successfully: $REASON"
	exit 1
    fi
done
