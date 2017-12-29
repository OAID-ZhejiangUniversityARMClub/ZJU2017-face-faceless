# -*- coding: utf-8 -*
import numpy as np  
import sys,os  
import cv2
caffe_root = '/media/scs4450/hard/luyi/util/caffe-ssd/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  
import time;  


net_file= 'm10-deploy.prototxt'   # faceboxes_deploy.prototxt, deploy-512.prototxt, m3_deploy.prototxt, deploy.prototxt
caffe_model='m10-deploy.caffemodel'   #train_faceboxes_umdface_m3_iter_20000.caffemodel, train_faceboxes_umdface_iter_180000.caffemodel, train_faceboxes_umdface_m2_iter_180000.caffemodel, train_faceboxes_umdface_iter_180000.caffemodel
test_dir = "images"

if not os.path.exists(caffe_model):
    print("FaceBoxes_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
caffe.set_mode_cpu()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ('background',
           'face')

'''transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
print net.blobs['data'].data.shape
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel'''
def img_process(img):
    ori_shape = img.shape
    bigger_side = ori_shape[0] if ori_shape[0] > ori_shape[1] else ori_shape[1]
    scale = 384. / bigger_side; size = (int(ori_shape[1]*scale), int(ori_shape[0]*scale))
    #scale = 324. / bigger_side; size = (int(ori_shape[1]*scale), int(ori_shape[0]*scale))
    ret = cv2.resize(img, size)
    return ret

def bord_img(img):
    BLACK = [0, 0, 0]
    shape = img.shape
    diff = abs(shape[0] - shape[1])
    if shape[0] < shape[1]:
        cons = cv2.copyMakeBorder(img, int(diff/2), int(diff-diff/2), 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    elif shape[0] > shape[1]:
        cons = cv2.copyMakeBorder(img, 0, 0, int(diff/2), int(diff-diff/2), cv2.BORDER_CONSTANT, value=BLACK)
    else:
        cons = img
    cons = cv2.resize(cons, (384, 384))
    return cons

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    print out['detection_out']
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
    #print box
    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    videoCapture = cv2.VideoCapture(imgfile)
    success, frame = videoCapture.read()
    border_img = img_process(frame)
    input_shape = (1, border_img.shape[2], border_img.shape[0], border_img.shape[1])
    transformer = caffe.io.Transformer({'data': input_shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    net.blobs['data'].reshape(1, border_img.shape[2], border_img.shape[0], border_img.shape[1])
    while success :
      success, frame = videoCapture.read()
      border_img = img_process(frame)
      transformed_image = transformer.preprocess('data', border_img)
      net.blobs['data'].data[...] = transformed_image


      time_start=time.time()
      out = net.forward()  
      time_end=time.time()
      print time_end-time_start,  
      print "s"

      box, conf, cls = postprocess(frame, out)

      for i in range(len(box)):
         p1 = (box[i][0], box[i][1])
         p2 = (box[i][2], box[i][3])
         cv2.rectangle(frame, p1, p2, (0,255,0))
         p3 = (max(p1[0], 15), max(p1[1], 15))
         title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
         cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
         cv2.imshow("SSD", frame)
     

      if cv2.waitKey(100) & 0xFF == ord('q'):
        break
      #Exit if ESC pressed
      #if k == 27 : return False
    print 'Done'
    return True

def detect_cap():
    videoCapture = cv2.VideoCapture(0)
    success, frame = videoCapture.read()
    border_img = img_process(frame)
    input_shape = (1, border_img.shape[2], border_img.shape[0], border_img.shape[1])
    transformer = caffe.io.Transformer({'data': input_shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    net.blobs['data'].reshape(1, border_img.shape[2], border_img.shape[0], border_img.shape[1])
    while success :
      success, frame = videoCapture.read()
      border_img = img_process(frame)
      transformed_image = transformer.preprocess('data', border_img)
      net.blobs['data'].data[...] = transformed_image


      time_start=time.time()
      out = net.forward()  
      time_end=time.time()
      print time_end-time_start, 's'

      box, conf, cls = postprocess(frame, out)

      for i in range(len(box)):
         p1 = (box[i][0], box[i][1])
         p2 = (box[i][2], box[i][3])
         cv2.rectangle(frame, p1, p2, (0,255,0))
         p3 = (max(p1[0], 15), max(p1[1], 15))
         title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
         cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
         cv2.imshow("SSD", frame)
     

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      #Exit if ESC pressed
      #if k == 27 : return False
    print 'Done'
    return True


def detect_file(imgfile):
    frame = cv2.imread(imgfile)
    #border_img = frame
    border_img = img_process(frame)
    input_shape = (1, border_img.shape[2], border_img.shape[0], border_img.shape[1])
    transformer = caffe.io.Transformer({'data': input_shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    net.blobs['data'].reshape(1, border_img.shape[2], border_img.shape[0], border_img.shape[1])
    transformed_image = transformer.preprocess('data', border_img)
    net.blobs['data'].data[...] = transformed_image

    count = 1
    while count < 100:
        time_start=time.time()
        out = net.forward()  
        time_end=time.time()
        count = count + 1
        print time_end-time_start,  
        print "s"

    box, conf, cls = postprocess(frame, out)

    for i in range(len(box)):
        p1 = (box[i][0], box[i][1])
        p2 = (box[i][2], box[i][3])
        cv2.rectangle(frame, p1, p2, (0,255,0))
        p3 = (max(p1[0], 15), max(p1[1], 15))
        title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
        cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (255, 0, 0), 1)
    #cv2.imshow("SSD", frame)
    cv2.imwrite('result.jpg', frame)
    #Exit if ESC pressed
    #if k == 27 : return False
    print 'Done'
    return True

'''for f in os.listdir(test_dir):
    if detect(test_dir + "/" + f) == False:
       break'''
if __name__ == "__main__":
    detect_file('test1.jpg')
    #detect('test.mp4')
    #detect_cap()