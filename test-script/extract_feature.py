# -*- coding: utf-8 -*
import numpy as np  
import sys,os  
import cv2
caffe_root = '/media/scs4450/hard/luyi/GPU/caffe-master/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  
import time;  


net_file= 'sphereface4-half_deploy_nobn.prototxt'   # sphereface4-half_deploy.prototxt
caffe_model='sphereface4-half_deploy_nobn.caffemodel'   # sphereface4half_model_iter_80000.caffemodel

if not os.path.exists(caffe_model):
    print("FaceBoxes_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
caffe.set_mode_cpu()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

def extract(imgfile1, imgfile2):
    frame = cv2.imread(imgfile1); #frame_1 = cv2.flip(frame, 1)
    frame = frame - 127.5; frame = frame / 128; img_t = frame.transpose((2,0,1))
    image = img_t.reshape((1, img_t.shape[0], img_t.shape[1], img_t.shape[2]))
    net.blobs['data'].data[...] = image
    out = net.forward() 
    f1 = out['fc5/sphere'][0,:].tolist()
    f = open('data1.txt', 'w')
    f.write(str(f1).replace('[','').replace(']','').replace(',',''))
    f.close()
    print f1

    frame = cv2.imread(imgfile2)
    frame = frame - 127.5; frame = frame / 128; img_t = frame.transpose((2,0,1))
    image = img_t.reshape((1, img_t.shape[0], img_t.shape[1], img_t.shape[2]))
    net.blobs['data'].data[...] = image
    out = net.forward() 
    f2 = out['fc5/sphere'][0,:].tolist()
    f = open('data2.txt', 'w')
    f.write(str(f2).replace('[','').replace(']','').replace(',',''))
    f.close()
    print f2
    '''img_t = frame_1.transpose((2,0,1)); img_t = img_t - 127.5; img_t = img_t / 128
    image = img_t.reshape((1, img_t.shape[0], img_t.shape[1], img_t.shape[2]))
    net.blobs['data'].data[...] = image
    out = net.forward() 
    f2 = out['fc5/sphere'][0,:].tolist()
    #print f2
    f1.extend(f2)
    print f1
    f = open('data1.txt', 'w')
    f.write(str(f1).replace('[','').replace(']','').replace(',',''))
    f.close()'''


if __name__ == "__main__":
    #detect_file('test3.jpg')
    #detect('test.mp4')
    #detect_cap()
    extract('/media/scs4450/hard/luyi/util/extract_feature/cjs/one/1-1-final.jpg', '/media/scs4450/hard/luyi/util/extract_feature/cjs/three/1-1-final.jpg')