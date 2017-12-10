#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV3 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inceptionv3_on_flowers.sh

HDFS_ENABLED=False
if [ "${HDFS_ENABLED}" == "True" ]; then
  # Where the pre-trained InceptionV3 checkpoint is saved to.
  PRETRAINED_CHECKPOINT_DIR=hdfs://namenode:9000/checkpoints
  # Where the training (fine-tuned) checkpoint and logs will be saved to.
  TRAIN_DIR=hdfs://namenode:9000/flowers-models/inception_v3
  # Where the dataset is saved to.
  DATASET_DIR=hdfs://namenode:9000/datasets/flowers
else
  # Where the pre-trained InceptionV3 checkpoint is saved to.
  PRETRAINED_CHECKPOINT_DIR=/tmp/checkpoints
  # Where the training (fine-tuned) checkpoint and logs will be saved to.
  TRAIN_DIR=/tmp/flowers-models/inception_v3
  # Where the dataset is saved to.
  DATASET_DIR=/tmp/flowers
fi

MAX_STEPS=1000
LOG_STEPS=100

# Checkpoint
SAVE_SECS=60
SAVE_STEPS=0

# Summary
SAVE_SUMMARIES_SECS=60

# Overwrite default values
ENV_PATH=./scripts/env.sh
if [ -e "${ENV_PATH}" ]; then
  source ${ENV_PATH}
fi

# Re-check the hdfs flag
if [ "${HDFS_ENABLED}" == "True" ]; then
  source ./hadoop_env.sh
fi

# Download the pre-trained checkpoint.
if [ "${HDFS_ENABLED}" == "True" ]; then
  echo "Warning: pass downloading the pre-trained checkpoint."
else
  if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
    mkdir ${PRETRAINED_CHECKPOINT_DIR}
  fi
  if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt ]; then
    wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    tar -xvf inception_v3_2016_08_28.tar.gz
    mv inception_v3.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt
    rm inception_v3_2016_08_28.tar.gz
  fi
fi

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

# Fine-tune only the new layers.
OPTS="--train_dir=${TRAIN_DIR} \
  --dataset_name=flowers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=${MAX_STEPS} \
  --batch_size=32 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=${SAVE_SECS} \
  --save_summaries_secs=${SAVE_SUMMARIES_SECS} \
  --log_every_n_steps=${LOG_STEPS} \
  --optimizer=rmsprop \
  --weight_decay=0.00004"

if [ "${HDFS_ENABLED}" == "True" ]; then
  CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob) python train_image_classifier.py ${OPTS}
else
  python train_image_classifier.py ${OPTS}
fi

# Run evaluation.
#EVAL_OPTS="--checkpoint_path=${TRAIN_DIR} \
  #--eval_dir=${TRAIN_DIR} \
  #--dataset_name=flowers \
  #--dataset_split_name=validation \
  #--dataset_dir=${DATASET_DIR} \
  #--model_name=inception_v3"

#if [ "${HDFS_ENABLED}" == "True" ]; then
  #CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob) python eval_image_classifier.py ${EVAL_OPTS}
#else
  #python eval_image_classifier.py ${EVAL_OPTS}
#fi


# Fine-tune all the new layers for 500 steps.
#python train_image_classifier.py \
#  --train_dir=${TRAIN_DIR}/all \
#  --dataset_name=flowers \
#  --dataset_split_name=train \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=inception_v3 \
#  --checkpoint_path=${TRAIN_DIR} \
#  --max_number_of_steps=500 \
#  --batch_size=32 \
#  --learning_rate=0.0001 \
#  --learning_rate_decay_type=fixed \
#  --save_interval_secs=60 \
#  --save_summaries_secs=60 \
#  --log_every_n_steps=10 \
#  --optimizer=rmsprop \
#  --weight_decay=0.00004
#
## Run evaluation.
#python eval_image_classifier.py \
#  --checkpoint_path=${TRAIN_DIR}/all \
#  --eval_dir=${TRAIN_DIR}/all \
#  --dataset_name=flowers \
#  --dataset_split_name=validation \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=inception_v3
