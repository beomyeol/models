#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV1 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./scripts/train_inception_v1_on_flowers.sh

HDFS_ENABLED=False
if [ "${HDFS_ENABLED}" == "True" ]; then
  # Where the training (fine-tuned) checkpoint and logs will be saved to.
  TRAIN_DIR=hdfs://namenode:9000/flowers-models/inception_v1
  DATASET_DIR=hdfs://namenode:9000/datasets/flowers
else
  # Where the training (fine-tuned) checkpoint and logs will be saved to.
  TRAIN_DIR=/tmp/flowers-models/inception_v1
  # Where the dataset is saved to.
  DATASET_DIR=/tmp/flowers
fi

MAX_STEPS=3000
LOG_STEPS=100

# Checkpoint
SAVE_SECS=0
SAVE_STEPS=0

# Summary
SAVE_SUMMARIES_SECS=0

# Overwrite default values
ENV_PATH=./scripts/env.sh
if [ -e "${ENV_PATH}" ]; then
  source ${ENV_PATH}
fi

# Re-check the hdfs flag
if [ "${HDFS_ENABLED}" == "True" ]; then
  source ./hadoop_env.sh
fi

OPTS="--train_dir=${TRAIN_DIR} \
  --dataset_name=flowers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v1 \
  --batch_size=32 \
  --save_interval_secs=${SAVE_SECS} \
  --save_steps=${SAVE_STEPS} \
  --save_summaries_secs=${SAVE_SUMMARIES_SECS} \
  --log_every_n_steps=${LOG_STEPS} \
  --optimizer=rmsprop \
  --weight_decay=0.00004 \
  --train_dir=${TRAIN_DIR} \
  --max_number_of_steps=${MAX_STEPS} \
  --learning_rate=0.01"

# Download the dataset
if [ "${HDFS_ENABLED}" == "True" ]; then
  CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob) python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir=${DATASET_DIR}
else
  python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir=${DATASET_DIR}
fi

# Run training.
if [ "${HDFS_ENABLED}" == "True" ]; then
  CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob) python train_image_classifier.py ${OPTS}
else
  python train_image_classifier.py ${OPTS}
fi
