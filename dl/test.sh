#!/bin/bash
./frcnn_gpu.bin params2 voc2007test.txt detections.bin
./eval.py detections.bin
