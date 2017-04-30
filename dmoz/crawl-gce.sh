#!/usr/bin/env bash
KEY=$(cat ~/.ssh/id_rsa.pub)
#INSTANCE_IP=$(gcloud compute --project "brave-monitor-160414" instances create "crawl-instance-1" --zone "europe-west1-c" \
#    --machine-type "n1-standard-2" \
#    --format "json" \
#    --subnet "default" \
#    --metadata "ssh-keys=ubuntu:${KEY}" \
#    --maintenance-policy "MIGRATE" \
#    --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring.write","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
#    --image "ubuntu-1604-xenial-v20170330" \
#    --image-project "ubuntu-os-cloud" \
#    --boot-disk-size "200" \
#    --boot-disk-type "pd-standard" \
#    --boot-disk-device-name "crawl-instance-1" | jq --raw-output ".[0].networkInterfaces[0].accessConfigs[0].natIP")
INSTANCE_IP="35.187.40.170"
echo ${INSTANCE_IP}

ssh ubuntu@${INSTANCE_IP} <<'ENDSSH'
sudo apt-get update && sudo apt-get -y upgrade
sudo apt-get -y install python-dev python-pip libxml2-dev libxslt1-dev zlib1g-dev libffi-dev libssl-dev
sudo pip install --upgrade pip
pip install scrapy
git clone https://github.com/fabianmurariu/website-categories-nn.git
cd website-categories-nn/dmoz
ENDSSH