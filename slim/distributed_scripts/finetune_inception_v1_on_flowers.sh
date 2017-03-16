#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV1 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_v1_on_flowers.sh

# Where the pre-trained InceptionV1 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/train_logs/flowers-models/inception_v1

# Where the dataset is saved to.
DATASET_DIR=/datasets/flowers

ONLY_NEW_MAX_STEPS=2000
ENTIRE_MAX_STEPS=1000

PS_HOSTS=172.17.0.2:2222
WORKER_HOSTS=172.17.0.1:2222

PS_ON_CPU=False

# Checkpoint
SAVE_SECS=60
SAVE_STEPS=0

# Summary
SAVE_SUMMARIES_SECS=120

USAGE="$0 [JOB NAME: ps or worker] [ID]"

JOB_NAME=$1
TASK_ID=$2

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

# Download the pre-trained checkpoint.

if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_v1.ckpt ]; then
  wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
  tar -xvf inception_v1_2016_08_28.tar.gz
  mv inception_v1.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_v1.ckpt
  rm inception_v1_2016_08_28.tar.gz
fi

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=flowers \
  --dataset_dir=${DATASET_DIR}

OPTS="--type=distributed \
  --job_name=${JOB_NAME} \
  --ps_hosts=${PS_HOSTS} \
  --worker_hosts=${WORKER_HOSTS} \
  --task=${TASK_ID} \
  --dataset_name=flowers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v1 \
  --batch_size=32 \
  --save_interval_secs=${SAVE_SECS} \
  --save_steps=${SAVE_STEPS} \
  --save_summaries_secs=${SAVE_SUMMARIES_SECS} \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 \
  --ps_on_cpu=${PS_ON_CPU}"

# Fine-tune only the new layers for 2000 steps.
ONLY_NEW_OPTS="--learning_rate=0.01 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v1.ckpt \
  --checkpoint_exclude_scopes=InceptionV1/Logits \
  --trainable_scopes=InceptionV1/Logits \
  --train_dir=${TRAIN_DIR} \
  --max_number_of_steps=${ONLY_NEW_MAX_STEPS} "

python train_image_classifier.py ${OPTS} ${ONLY_NEW_OPTS}

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v1

# Fine-tune all the new layers for 1000 steps.
ENTIRE_OPTS="--train_dir=${TRAIN_DIR}/all \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=${ENTIRE_MAX_STEPS} \
  --learning_rate=0.001"

python train_image_classifier.py ${OPTS} ${ENTIRE_OPTS}

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v1
