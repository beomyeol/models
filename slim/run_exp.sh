#!/bin/bash

VERSION=3

TYPE=localFS
#TYPE=hdfs

if [ "$TYPE" == "localFS" ]; then
  WORK_DIR=/mnt/extra
else
  WORK_DIR=hdfs://node-1:9000/user/beomyeol
fi

# Where the pre-trained Inception checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=${WORK_DIR}/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
#TRAIN_DIR=/tmp/flowers-models/inception_v${VERSION}
TRAIN_DIR=${WORK_DIR}/flowers-models/inception_v${VERSION}

# Where the dataset is saved to.
DATASET_DIR=${WORK_DIR}/flowers

# Checkpoint
SAVE_SECS=0
SAVE_STEPS=0

# Summary
SAVE_SUMMARIES_SECS=0

BATCH_SIZE=32

MAX_STEPS=30

if [ "$TYPE" == "localFS" ]; then
  # Download the pre-trained checkpoint.
  if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
    mkdir ${PRETRAINED_CHECKPOINT_DIR}
  fi
  if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_v${VERSION}.ckpt ]; then
    wget http://download.tensorflow.org/models/inception_v${VERSION}_2016_08_28.tar.gz
    tar -xvf inception_v${VERSION}_2016_08_28.tar.gz
    mv inception_v${VERSION}.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_v${VERSION}.ckpt
    rm inception_v${VERSION}_2016_08_28.tar.gz
  fi
else
  echo 'Check files in HDFS and comment out exit code'
  exit
fi

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=flowers \
  --dataset_dir=${DATASET_DIR}

if [ "$1" == "" ]; then
  echo "Id must be given"
  exit
fi

# Log directory
LOG_DIR=~/ml/tf/logs/inception_v${VERSION}/${TYPE}/${BATCH_SIZE}-batch/$1

if [ ! -d "${LOG_DIR}" ]; then
  mkdir -p ${LOG_DIR}
fi

if [ "${VERSION}" == "1" ]; then
  CKPT_EXCLUDE_SCOPES="InceptionV1/Logits"
  TRAINABLE_SCOPES="InceptionV1/Logits"
else
  CKPT_EXCLUDE_SCOPES="InceptionV3/Logits,InceptionV3/AuxLogits"
  TRAINABLE_SCOPES="InceptionV3/Logits,InceptionV3/AuxLogits"
fi

#declare -a arr=(1 10 20 50 70 100)
#declare -a arr=(0 10 30 60 120 180)
declare -a arr=(0)
for i in "${arr[@]}"
do
  SAVE_STEPS=$i
  LOG_FILENAME="ckpt-${i}step.log"
  #SAVE_SECS=$i
  #LOG_FILENAME="ckpt-${i}secs.log"
  if [ "$TYPE" == "localFS" ]; then
    rm -r ${TRAIN_DIR}/*
  else
    hdfs dfs -rm ${TRAIN_DIR}/*
    CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob)
  fi
  python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=inception_v${VERSION} \
    --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v${VERSION}.ckpt \
    --checkpoint_exclude_scopes=${CKPT_EXCLUDE_SCOPES} \
    --trainable_scopes=${TRAINABLE_SCOPES} \
    --max_number_of_steps=${MAX_STEPS} \
    --batch_size=${BATCH_SIZE} \
    --learning_rate=0.01 \
    --learning_rate_decay_type=fixed \
    --save_interval_secs=${SAVE_SECS} \
    --save_steps=${SAVE_STEPS} \
    --save_summaries_secs=${SAVE_SUMMARIES_SECS} \
    --log_every_n_steps=10 \
    --optimizer=rmsprop \
    --weight_decay=0.00004 |& tee /tmp/${LOG_FILENAME}
  mv /tmp/${LOG_FILENAME} ${LOG_DIR}
done


