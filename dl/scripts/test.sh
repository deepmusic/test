#!/bin/bash
./demo_gpu.bin database ../test/temp scripts/voc2007test.txt detections.bin
./scripts/eval.py detections.bin
