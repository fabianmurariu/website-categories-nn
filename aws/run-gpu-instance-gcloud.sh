#!/usr/bin/env bash
set -x
set -e

# copy provision-docker-gpu.sh to gs
gsutil cp provision-docker-gpu.sh gs://code-deploy/provision-docker-gpu.sh

# Create the instance with the GPU
PROJECT="brave-monitor-160414"
KEY=$(cat ~/.ssh/id_rsa.pub)
gcloud compute --project ${PROJECT} instances create "instance-1" --zone "europe-west1-b" \
    --machine-type "n1-highmem-8" \
    --subnet "default" \
    --metadata "Name=gpu-deep-learning-1,ssh-keys=ubuntu:${KEY}" \
    --maintenance-policy "TERMINATE" \
    --service-account "844526962052-compute@developer.gserviceaccount.com" \
    --scopes "https://www.googleapis.com/auth/cloud-platform" \
    --accelerator type=nvidia-tesla-k80,count=1 \
    --min-cpu-platform "Automatic" \
    --image "ubuntu-1604-xenial-v20170803" \
    --image-project "ubuntu-os-cloud" \
    --boot-disk-size "15" \
    --boot-disk-type "pd-standard" \
    --boot-disk-device-name "instance-1"

# take down the instance
gcloud compute ssh ubuntu@instance-1 --command 'sudo shutdown -h now'

# create the image
gcloud compute --project=brave-monitor-160414 images create deep-learning-gpu-image-1 --source-disk=instance-1 --source-disk-zone=europe-west1-b