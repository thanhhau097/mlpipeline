#!/bin/bash

full_image_name=thanhhau097/ocrdeployment:v1.0

docker build -t "${full_image_name}" .
docker push "$full_image_name"