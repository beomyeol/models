#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Cifar10 dataset
# 2. Trains a CifarNet model on the Cifar10 training set.
# 3. Evaluates the model on the Cifar10 testing set.
#
# Usage:
# cd slim
# ./distributed_scripts/train_cifar_net_on_mnist.sh

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/cifarnet-model

# Where the dataset is saved to.
DATASET_DIR=/tmp/cifar10

MAX_STEPS=100000

# Checkpoint
SAVE_SECS=120
SAVE_STEPS=0

# Summary
SAVE_SUMMARIES_SECS=120

# Options for distributed training
USAGE="$0 [JOB NAME: ps or worker] [ID]"

JOB_NAME=$1
TASK_ID=$2

PS_HOSTS=localhost:2222
WORKER_HOSTS=localhost:2223
PS_ON_CPU=True

# Overwrite default values
ENV_PATH=./distributed_scripts/env.sh
if [ -e "${ENV_PATH}" ]; then
  source ${ENV_PATH}
fi

if [ ! "${JOB_NAME}" == "ps" ] && [ ! "${JOB_NAME}" == "worker" ]; then
  echo "ERROR: Invalid job name: ${JOB_NAME}"
  echo $USAGE
  exit
fi

re='^[0-9]+$'
if ! [[ ${TASK_ID} =~ $re ]]; then
  echo "ERROR: Invalid id: ${TASK_ID}"
  echo $USAGE
  exit
fi

OPTS="--type=distributed \
  --job_name=${JOB_NAME} \
  --ps_hosts=${PS_HOSTS} \
  --worker_hosts=${WORKER_HOSTS} \
  --task=${TASK_ID} \
  --train_dir=${TRAIN_DIR}"

if [ "${JOB_NAME}" == "ps" ]; then # PS
  OPTS+=" --ps_on_cpu=${PS_ON_CPU}"
else # Worker
  # Download the dataset
  python download_and_convert_data.py \
    --dataset_name=cifar10 \
    --dataset_dir=${DATASET_DIR}

  OPTS+=" --dataset_name=cifar10 \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=cifarnet \
    --preprocessing_name=cifarnet \
    --max_number_of_steps=${MAX_STEPS} \
    --batch_size=128 \
    --save_interval_secs=${SAVE_SECS} \
    --save_steps=${SAVE_STEPS} \
    --save_summaries_secs=${SAVE_SUMMARIES_SECS} \
    --log_every_n_steps=100 \
    --optimizer=sgd \
    --learning_rate=0.1 \
    --learning_rate_decay_factor=0.1 \
    --num_epochs_per_decay=200 \
    --weight_decay=0.004"
fi

# Run training.
python train_image_classifier.py ${OPTS}