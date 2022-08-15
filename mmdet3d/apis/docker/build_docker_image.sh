#!/usr/bin/env bash
#
# Script to build docker image with model conversion environment
#

DOCKER_IMAGE_TAG="model-conversion"
CUDNN8_VERSION=8.2.2.*
CUBLAS10_VERSION=10.2.3.*
TRT8_VERSION=8.2.3-1+cuda10.2

cd $(dirname $(readlink -f "$0"))
docker build -f Dockerfile-x86-tensorrt \
  --build-arg CUDNN8_VERSION=${CUDNN8_VERSION} \
  --build-arg CUBLAS10_VERSION=${CUBLAS10_VERSION} \
  --build-arg TRT8_VERSION=${TRT8_VERSION} \
  -t $DOCKER_IMAGE_TAG .

