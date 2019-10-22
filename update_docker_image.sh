#!/usr/bin/env bash

docker build -t motorbike-generator .
docker tag  motorbike-generator phucbui/motorbike-generator
docker push phucbui/motorbike-generator