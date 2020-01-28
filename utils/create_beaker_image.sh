#!/bin/bash

# Create a Beaker image using provided image name tagged with the current username and git commit hash
# of the repo.
# Usage: ./utils/create_beaker_image.sh image_name dockerfile
# Example: ./utils/beaker/create_beaker_image.sh launchpadqa Dockerfile

set -e

IMAGE_NAME=$1
DOCKERFILE_NAME=$2

# Image name
GIT_HASH=`git log --format="%h" -n 1`
IMAGE=${IMAGE_NAME}_$USER-$GIT_HASH
IM_NAME=$IMAGE

# Build the image (if needed)
if [[ "$(docker images -q $IMAGE 2> /dev/null)" == "" ]]; then
  echo "Building $IMAGE"
  docker build -f $DOCKERFILE_NAME -t $IMAGE .
  beaker image create --name=$IM_NAME --desc="Skidls Repo; Git Hash: $GIT_HASH" $IMAGE
else
  image_spec=`beaker image inspect $IM_NAME`
  if [[ -z $image_spec ]]; then
    echo "No beaker image with name $image_spec"
    unset $IM_NAME
    exit 1
  else
    echo "Running with beaker image: $image_spec"
  fi
fi
