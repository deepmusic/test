#!/bin/bash
./frcnn_gpu.bin voc2007test.txt detections.bin
./eval.py detections.bin
