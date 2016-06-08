#!/bin/bash
./demo_gpu.bin database scripts/params3 scripts/voc2007test.txt detections.bin
./scripts/eval.py detections.bin
