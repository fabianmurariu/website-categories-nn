#!/usr/bin/env bash

set -e

CLUSTER_NAME=${1?"Cluster Name required"}
KEY_NAME=${2?"KeyName required"}
SUBNET_ID=${3?"SubnetId required"}
EMR_MASTER_SG=${4?"EMR Master Security Group required"}
EMR_SLAVE_SG=${5?"EMR Slave Security Group required"}
INSTANCE_PROFILE=${6?"EMR Instance Profile"}

EC2_ATTRIBUTES=$(jq -n -c \
--arg KEY_NAME "$KEY_NAME" \
--arg SUBNET_ID "$SUBNET_ID" \
--arg EMR_MASTER_SG "$EMR_MASTER_SG" \
--arg EMR_SLAVE_SG "$EMR_SLAVE_SG" \
--arg INSTANCE_PROFILE "$INSTANCE_PROFILE" \
'{"KeyName":$KEY_NAME,"InstanceProfile":$INSTANCE_PROFILE,"SubnetId":$SUBNET_ID,"EmrManagedSlaveSecurityGroup":$EMR_MASTER_SG,"EmrManagedMasterSecurityGroup":$EMR_SLAVE_SG}')

CLUSTERID=$(aws emr create-cluster \
                --name "$CLUSTER_NAME" \
                 --tags "Name=$CLUSTER_NAME" \
                 --release-label emr-5.6.0 \
                 --applications Name=Ganglia Name=Hadoop Name=Hue Name=Spark \
                 --ec2-attributes ${EC2_ATTRIBUTES} \
                 --service-role EMR_DefaultRole \
                 --enable-debugging \
                 --log-uri 's3n://aws-logs-386296384021-eu-west-1/elasticmapreduce/' \
                 --instance-groups '[{"InstanceCount":1,"InstanceGroupType":"MASTER","InstanceType":"m4.2xlarge","Name":"Master instance group","BidPrice":"0.20"},{"InstanceCount":4,"InstanceGroupType":"CORE","InstanceType":"m4.2xlarge","Name":"Core instance group","BidPrice":"0.20"}]' \
                 --visible-to-all-users | grep -o 'j-\w*')
echo ${CLUSTERID}
