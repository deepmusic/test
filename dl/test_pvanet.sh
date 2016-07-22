#!/bin/bash

GPU=true
LIGHT_NET=true
COMPRESSION=true
PRE_NMS_TOPN=6000
POST_NMS_TOPN=200
INPUT_SCALE=576

MODEL_NAME=""
if [ ${GPU} == true ]; then
  MODEL_NAME+="_gpu"
else
  MODEL_NAME+="_cpu"
fi
if [ ${LIGHT_NET} == true ]; then
  MODEL_NAME+="_light"
fi
if [ ${COMPRESSION} == true ]; then
  MODEL_NAME+="_compress"
fi
echo ${MODEL_NAME}

./demo${MODEL_NAME}.bin ${PRE_NMS_TOPN} ${POST_NMS_TOPN} ${INPUT_SCALE} database data/voc2007test_small.txt detections.bin
