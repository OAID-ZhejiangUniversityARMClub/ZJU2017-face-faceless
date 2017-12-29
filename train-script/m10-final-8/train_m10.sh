./build/tools/caffe train \
--solver=examples/m10-final-8/solver.prototxt \
--weights=examples/m10-final-8/m10-final-7_324x324_384_12-24-48-96-162.caffemodel \
--gpu=1 2>&1 | tee examples/m10-final-8/logs/train_faceboxes_umdfaces_20171211.logs