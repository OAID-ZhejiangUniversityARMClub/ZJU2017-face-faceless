import numpy as np  
import sys,os 
import cv2 
caffe_root = '/media/scs4450/hard/luyi/util/caffe-ssd/'
sys.path.insert(0, caffe_root + 'python')  
import caffe 

train_proto = 'm10-final-3_324x324_norm_32-40-48-96-144.prototxt'  
train_model = 'm10-final-3_324x324_norm_32-40-48-96-144.caffemodel'  #should be your snapshot caffemodel

nobn_proto  = 'm10-final-3_324x324_norm_32-40-48-96-144_nobn.prototxt'
nobn_model  = 'm10-final-3_324x324_norm_32-40-48-96-144_nobn.caffemodel'

def merge_bn():
    net      = caffe.Net(train_proto, train_model, caffe.TEST)
    net_nobn = caffe.Net(nobn_proto, caffe.TEST)
    merge_dicts = { 'conv1':['conv1', 'conv1/bn', 'conv1/scale'],
                    'conv2-1':['conv2-1', 'conv2-1/bn', 'conv2-1/scale'],
                    'conv2-2':['conv2-2', 'conv2-2/bn', 'conv2-2/scale'],
                    'conv3/incep0/conv':['conv3/incep0/conv', 'conv3/incep0/bn', 'conv3/incep0/bn_scale'],
                    'conv3/incep1/conv1':['conv3/incep1/conv1', 'conv3/incep1/bn1', 'conv3/incep1/bn_scale1'],
                    'conv3/incep1/conv2':['conv3/incep1/conv2', 'conv3/incep1/bn2', 'conv3/incep1/bn_scale2'],
                    'conv3/incep2/conv1':['conv3/incep2/conv1', 'conv3/incep2/bn1', 'conv3/incep2/bn_scale1'],
                    'conv3/incep2/conv2':['conv3/incep2/conv2', 'conv3/incep2/bn2', 'conv3/incep2/bn_scale2'],
                    'conv3/incep2/conv3':['conv3/incep2/conv3', 'conv3/incep2/bn3', 'conv3/incep2/bn_scale3'],
                    'conv5/incep0/conv':['conv5/incep0/conv', 'conv5/incep0/bn', 'conv5/incep0/bn_scale'],
                    'conv5/incep1/conv1':['conv5/incep1/conv1', 'conv5/incep1/bn1', 'conv5/incep1/bn_scale1'],
                    'conv5/incep1/conv2':['conv5/incep1/conv2', 'conv5/incep1/bn2', 'conv5/incep1/bn_scale2'],
                    'conv5/incep2/conv1':['conv5/incep2/conv1', 'conv5/incep2/bn1', 'conv5/incep2/bn_scale1'],
                    'conv5/incep2/conv2':['conv5/incep2/conv2', 'conv5/incep2/bn2', 'conv5/incep2/bn_scale2'],
                    'conv5/incep2/conv3':['conv5/incep2/conv3', 'conv5/incep2/bn3', 'conv5/incep2/bn_scale3'],
                    'conv6_1':['conv6_1', 'conv6/bn1', 'conv6/bn_scale1'],
                    'conv6_2':['conv6_2', 'conv6/bn2', 'conv6/bn_scale2'],
                    'conv7_1':['conv7_1', 'conv7/bn1', 'conv7/bn_scale1'],
                    'conv7_2':['conv7_2', 'conv7/bn2', 'conv7/bn_scale2']
    }
    no_merge_dicts = [  'Inception3/conv/loc1', 'Inception3/conv/conf1', 'Inception3/conv/loc2', 'Inception3/conv/conf2', 
                        'Inception3/conv/loc3', 'Inception3/conv/conf3', 'conv6/loc', 'conv6/conf', 'conv7/loc', 'conv7/conf']
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
    transformer      = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer_onbn = caffe.io.Transformer({'data': net_nobn.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer_onbn.set_transpose('data', (2,0,1))
    img = cv2.imread('test10x10.jpg')
    transformed_image = transformer.preprocess('data', img)
    transformed_nobn_image = transformer.preprocess('data', img)
    net.blobs['data'].data[...] = transformed_image
    net_nobn.blobs['data'].data[...] = transformed_nobn_image

    caffe.set_mode_gpu()
    out = net.forward(); out_nobn = net_nobn.forward()
    print out['conv1'][0, 0, :, :]
    print out_nobn['conv1'][0, 0, :, :]

if __name__ == '__main__':
    merge_bn()
    #test_model()

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