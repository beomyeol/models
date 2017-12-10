#!/bin/bash

HDFS_ENABLED=True

#TRAIN_DIR=hdfs://node-1:9000/user/beomyeol/cifarnet-model
#DATASET_DIR=hdfs://node-1:9000/user/beomyeol/datasets/cifar10

#TRAIN_DIR=hdfs://node-1:9000/user/beomyeol/flowers-models/inception_v3
#DATASET_DIR=hdfs://node-1:9000/user/beomyeol/datasets/flowers
PRETRAINED_CHECKPOINT_DIR=hdfs://node-1:9000/user/beomyeol/checkpoints

TRAIN_DIR=hdfs://node-1:9000/user/beomyeol/flowers-models/resnet_v1_50
DATASET_DIR=hdfs://node-1:9000/user/beomyeol/datasets/flowers

MAX_STEPS=150
LOG_STEPS=100

SAVE_SECS=60
SAVE_STEPS=0
SAVE_SUMMARIES_SECS=0

