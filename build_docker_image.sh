#!/usr/bin/env bash

set -e

IMAGE_NAME=modularqa
DOCKERFILE_NAME=Dockerfile

# Image name
GIT_HASH=`git log --format="%h" -n 1`
IMAGE=$IMAGE_NAME_$USER-$GIT_HASH

docker build -f $DOCKERFILE_NAME -t $IMAGE .

echo "Built image $IMAGE. Now run:"
echo "beaker image create --name=$IMAGE --desc=\"Transformers:ModularQA Repo; Git Hash: $GIT_HASH\" $IMAGE"