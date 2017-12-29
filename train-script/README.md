# Face_A7

## 子文件夹说明
用来模型训练的相关文件，faceboxes项目来自[zeusees/FaceBoxes
](https://github.com/zeusees/FaceBoxes)，sphereface项目来自[wy1iu/sphereface
](https://github.com/wy1iu/sphereface)

主要有以下几个文件
- m10-final-8：现阶段用于检测的模型相关文件，包含了merge_bn.py文件用来合并批归一化层
- sphere-4-half：现阶段用于提取特征的模型相关文件，包含了merge_bn.py文件用来合并批归一化层
- transform_umdfaces.py：将umdfaces转换成SSD训练数据的形式