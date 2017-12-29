#!/bin/bash
# Usage:
# ./code/sphereface_train.sh GPU
#
# Example:
# ./code/sphereface_train.sh 0,1,2,3

./build/tools/caffe train -solver=models/sphere-4-half/sphereface4-half_solver.prototxt -weights=models/sphere-4-half/sphere-4-half_pretrain__iter_500000.caffemodel -gpu=0,1 2>&1 | tee models/sphere-4-half/logs/sphereface4-half_train-20171104.log