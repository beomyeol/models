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
  # Where the training (fine-tuned) checkpoint and logs will be saved to.
  TRAIN_DIR=hdfs://namenode:9000/flowers-models/inception_v3
  DATASET_DIR=hdfs://namenode:9000/datasets/flowers
else
  # Where the training (fine-tuned) checkpoint and logs will be saved to.
  TRAIN_DIR=/tmp/flowers-models/inception_v3
  # Where the dataset is saved to.
  DATASET_DIR=/tmp/flowers
fi

MAX_STEPS=3000
LOG_STEPS=10

# Checkpoint
SAVE_SECS=0
SAVE_STEPS=0

# Summary
SAVE_SUMMARIES_SECS=0

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

# Re-check the hdfs flag
if [ "${HDFS_ENABLED}" == "True" ]; then
  source ./distributed_scripts/hadoop_env.sh
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
  if [ "${HDFS_ENABLED}" == "True" ]; then
    CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob) python download_and_convert_data.py \
      --dataset_name=flowers \
      --dataset_dir=${DATASET_DIR}
  else
    python download_and_convert_data.py \
      --dataset_name=flowers \
      --dataset_dir=${DATASET_DIR}
  fi

  OPTS+=" --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=inception_v3 \
    --max_number_of_steps=${MAX_STEPS} \
    --batch_size=32 \
    --learning_rate=0.001 \
    --learning_rate_decay_type=fixed \
    --save_interval_secs=${SAVE_SECS} \
    --save_steps=${SAVE_STEPS} \
    --save_summaries_secs=${SAVE_SUMMARIES_SECS} \
    --log_every_n_steps=${LOG_STEPS} \
    --optimizer=rmsprop \
    --weight_decay=0.00004"
fi

# Run training.
if [ "${HDFS_ENABLED}" == "True" ]; then
  CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob) python train_image_classifier.py ${OPTS}
else
  python train_image_classifier.py ${OPTS}
fi
