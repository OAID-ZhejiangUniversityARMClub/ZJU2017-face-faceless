import numpy as np  
import sys,os 
import cv2 
caffe_root = '/media/scs4450/hard/luyi/util/caffe-ssd/'
sys.path.insert(0, caffe_root + 'python')  
import caffe 

train_proto = 'spherefacem2-sgd_deploy.prototxt'  
train_model = 'sphere-m2-sgd_train_iter_260000.caffemodel'  #should be your snapshot caffemodel

nobn_proto  = 'sphereface_128_nobn.prototxt'
nobn_model  = 'sphereface_128_nobn.caffemodel'

def merge_bn():
    net      = caffe.Net(train_proto, train_model, caffe.TEST)
    net_nobn = caffe.Net(nobn_proto, caffe.TEST)
    merge_dicts = { 'conv1_1':['conv1_1', 'conv1_1/bn', 'conv1_1/scale'],
                    'conv2_1':['conv2_1', 'conv2_1/bn', 'conv2_1/scale'],
                    'conv3_1':['conv3_1', 'conv3_1/bn', 'conv3_1/scale'],
                    'conv4_1':['conv4_1', 'conv4_1/bn', 'conv4_1/scale']
    }
    prelu_dicts = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']
    no_merge_dicts = ['fc5/sphere']
    for key in prelu_dicts:
        w = net.params[key][0].data
        net_nobn.params[key][0].data[...] = w

    for key in no_merge_dicts:
        conv_w = net.params[key][0].data
        n      = conv_w.shape[0]
        conv_b = np.zeros(n)
        if len(net.params[key]) > 0:
            conv_b = net.params[key][1].data
        net_nobn.params[key][0].data[...] = conv_w
        net_nobn.params[key][1].data[...] = conv_b

    for key, value in merge_dicts.iteritems():
        conv_w = net.params[value[0]][0].data
        n      = conv_w.shape[0]
        conv_b = np.zeros(n)
        if len(net.params[value[0]]) > 0:
            conv_b = net.params[value[0]][1].data

        bn_mean = net.params[value[1]][0].data
        bn_var  = net.params[value[1]][1].data
        bn_fact = net.params[value[1]][2].data
        if bn_fact != 0:
            bn_fact = 1.0 / bn_fact
        bn_mean = bn_mean * bn_fact
        bn_var  = bn_var  * bn_fact
        bn_std  = np.sqrt(bn_var + 1e-11)
        std     = bn_std.reshape((n, 1, 1, 1))

        scale_alpha = net.params[value[2]][0].data
        scale_beta  = net.params[value[2]][1].data
        alpha       = scale_alpha.reshape((n, 1, 1, 1))

        w = conv_w * alpha / std
        b = (conv_b - bn_mean) * scale_alpha / bn_std + scale_beta
        net_nobn.params[key][0].data[...] = w
        net_nobn.params[key][1].data[...] = b
    
    net_nobn.save(nobn_model)
    print 'Done!'

def test_model():
    net      = caffe.Net(train_proto, train_model, caffe.TEST)
    net_nobn = caffe.Net(nobn_proto, nobn_model, caffe.TEST)
    frame = cv2.imread('Abel_Pacheco_0001.jpg'); frame_1 = cv2.imread('Abel_Pacheco_0001.jpg')

    frame = frame - 127.5; frame = frame / 128; img_t = frame.transpose((2,0,1))
    image = img_t.reshape((1, img_t.shape[0], img_t.shape[1], img_t.shape[2]))
    frame_1 = frame_1 - 127.5; frame_1 = frame_1 / 128; img_t_1 = frame_1.transpose((2,0,1))
    image_1 = img_t_1.reshape((1, img_t_1.shape[0], img_t_1.shape[1], img_t_1.shape[2]))
    net.blobs['data'].data[...] = image
    net_nobn.blobs['data'].data[...] = image_1

    caffe.set_mode_gpu()
    out = net.forward(); out_nobn = net_nobn.forward()
    print out['fc5/sphere']
    print out_nobn['fc5/sphere']

if __name__ == '__main__':
    merge_bn()
    test_model()

'''for key in net.params.iterkeys():
    print key'''

'''conv1 = net.params['conv1']
print conv1[0].data.shape
print conv1[0].data

print conv1[1].data.shape
print conv1[1].data'''

'''conv1_bn = net.params['conv1/bn']
print conv1_bn[0].data.shape
print conv1_bn[0].data
print conv1_bn[1].data.shape
print conv1_bn[1].data
print conv1_bn[2].data.shape
print conv1_bn[2].data'''

'''conv1_scale = net.params['conv1/scale']
print conv1_scale[0].data.shape
print conv1_scale[0].data
print conv1_scale[1].data.shape
print conv1_scale[1].data'''