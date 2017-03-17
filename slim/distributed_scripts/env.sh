#!/bin/bash

MACHINE_1=node-1
MACHINE_2=node-2
MACHINE_3=node-3
MACHINE_4=node-4

# MAX_STEPS=3000

# 1 PS and 3 Workers on distinct machines
# PS_HOSTS=${MACHINE_1}:2222
# WORKER_HOSTS=${MACHINE_2}:2223,${MACHINE_3}:2223,${MACHINE_4}:2223
# PS_ON_CPU=False

# Colocation of 4 PSs and 4 Workers on each machine
#PS_HOSTS=${MACHINE_1}:2222,${MACHINE_2}:2222,${MACHINE_3}:2222,${MACHINE_4}:2222
#WORKER_HOSTS=${MACHINE_1}:2223,${MACHINE_2}:2223,${MACHINE_3}:2223,${MACHINE_4}:2223
#PS_ON_CPU=True

# Colocation of 2 PSs and 4 Workers
#PS_HOSTS=${MACHINE_1}:2222,${MACHINE_2}:2222
#WORKER_HOSTS=${MACHINE_1}:2223,${MACHINE_2}:2223,${MACHINE_3}:2223,${MACHINE_4}:2223
#PS_ON_CPU=True

# Paths for Docker containers
#BASE_TRAIN_DIR=/train_logs
#BASE_DATASET_DIR=/datasets
#PRETRAINED_CHECKPOINT_DIR=/checkpoints

# Paths for inception_v1
#TRAIN_DIR=${BASE_TRAIN_DIR}/flowers-models/inception_v1
#DATASET_DIR=${BASE_DATASET_DIR}/flowers

# Paths for inception_v3
#TRAIN_DIR=${BASE_TRAIN_DIR}/flowers-models/inception_v3
#DATASET_DIR=${BASE_DATASET_DIR}/flowers

# Paths for CIFARNet
#TRAIN_DIR=${BASE_TRAIN_DIR}/cifarnet-model
#DATASET_DIR=${BASE_DATASET_DIR}/cifar10