#!/bin/bash

full_image_name=thanhhau097/ocrpreprocess

docker build -t "${full_image_name}" .
docker push "$full_image_name"