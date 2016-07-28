#!/bin/bash
CAFFE_DIR=../../caffe
TEMP_FILENAME=$2.temp.txt
${CAFFE_DIR}/build/tools/caffe test -gpu 1 -model $1 -weights $2 -iterations 1000 2>&1 | tee | grep ", dist = " > ${TEMP_FILENAME}
python lfw/process_face_output.py ${TEMP_FILENAME}
rm -f ${TEMP_FILENAME}
