#! /bin/bash

sudo docker run --rm -it --runtime=nvidia --name=$2 --ipc=host --user="$(id -u):$(id -g)" -v=/data:/data -v=/home/jonghwan/research:/home/jonghwan/research -w=/home/jonghwan/research/temporal_language_grounding choco1916/envs:temporal_grounding $1
