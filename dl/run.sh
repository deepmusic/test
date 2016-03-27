#!/bin/bash
./frcnn_cpu.bin list.txt detections.bin
./eval.py detections.bin
