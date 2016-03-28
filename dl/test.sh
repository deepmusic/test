#!/bin/bash
./frcnn_gpu.bin list.txt detections.bin
./eval.py detections.bin
