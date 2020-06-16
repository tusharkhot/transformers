#!/usr/bin/env bash

set -e

IMAGE_NAME=modularqa
DOCKERFILE_NAME=Dockerfile

# Image name
GIT_HASH=`git log --format="%h" -n 1`
IMAGE=$IMAGE_NAME_$USER-$GIT_HASH

docker build -f $DOCKERFILE_NAME -t $IMAGE .

echo "\033[0;32m Built image $IMAGE. Now run: \033[0m"
echo "\033[0;35m beaker image create --name=$IMAGE --desc=\"Transformers:ModularQA Repo; Git Hash: $GIT_HASH\" $IMAGE \033[0m"