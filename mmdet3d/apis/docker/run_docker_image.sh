#!/usr/bin/env bash
#
# Script to run docker image with model conversion environment in interactive mode.
# Note:
#   mmdetection3d repo will be mounted as /root/mmdetection3d
#   ~/checkpoints will be mounted as /root/checkpoints
#   ~/model_export repo will be mounted as /root/model_export
#

DOCKER_IMAGE="model-conversion:latest"
REPO_ROOT=$(dirname $(dirname $(dirname $(dirname $(readlink -f "$0")))))

docker run -it --rm --gpus all --name model-conversion\
           -v $REPO_ROOT:/root/mmdetection3d \
           -v ~/checkpoints:/root/checkpoints \
           -v ~/model_export:/root/model_export \
           $DOCKER_IMAGE
