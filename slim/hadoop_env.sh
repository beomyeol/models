#!/bin/bash

if [ "${JAVA_HOME}" == "" ]; then
  echo "JAVA_HOME must be set"
  exit 1
fi
if [ "${HADOOP_HOME}" == "" ]; then
  echo "HADOOP_HOME must be set"
  exit 1
fi

source ${HADOOP_HOME}/libexec/hadoop-config.sh
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${JAVA_HOME}/jre/lib/amd64/server

