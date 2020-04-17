#! /bin/bash

options=$1	# path to configuration file without ".yml"
m_type=$2	# model type
dataset=$3	# dataset type

CUDA_VISIBLE_DEVICES=$4 python -m src.experiment.eval \
		--exp ${options} \
		--model_type ${m_type} \
		--dataset ${dataset} \
		--start_epoch $5 \
		--end_epoch $6 \
		--epoch_stride $7 \
		--num_workers $8
