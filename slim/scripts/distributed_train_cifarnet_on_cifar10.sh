#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Cifar10 dataset
# 2. Trains a CifarNet model on the Cifar10 training set.
# 3. Evaluates the model on the Cifar10 testing set.
#
# Usage:
# cd slim
# ./scripts/train_cifar_net_on_mnist.sh

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

# Hosts
PS_HOSTS=172.17.0.2:2222
WORKER_HOSTS=172.17.0.1:2222

USAGE="$0 [JOB NAME: ps or worker] [ID]"

if [ ! "$1" == "ps" ] && [ ! "$1" == "worker" ]; then
  echo "ERROR: Invalid job name: $1"
  echo $USAGE
  exit
fi

re='^[0-9]+$'
if ! [[ $2 =~ $re ]]; then
  echo "ERROR: Invalid id: $2"
  echo $USAGE
  exit
fi

# Download the dataset
if [ "$1" == "worker" ]; then
  python download_and_convert_data.py \
    --dataset_name=cifar10 \
    --dataset_dir=${DATASET_DIR}
fi

# Run training.
python train_image_classifier.py \
  --type=distributed \
  --job_name=$1 \
  --ps_hosts=${PS_HOSTS} \
  --worker_hosts=${WORKER_HOSTS} \
  --task=$2 \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cifar10 \
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
  --weight_decay=0.004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=cifar10 \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=cifarnet
