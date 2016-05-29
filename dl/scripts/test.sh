#!/bin/bash
./frcnn_gpu.bin scripts/params scripts/voc2007test.txt detections.bin
./scripts/eval.py detections.bin
