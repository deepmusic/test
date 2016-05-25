#!/bin/bash
./frcnn_gpu.bin params voc2007test.txt detections.bin
./eval.py detections.bin
