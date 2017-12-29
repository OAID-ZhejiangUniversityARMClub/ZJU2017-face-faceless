#!/usr/bin/env python

from tool_csv import loadCSVFile
import cv2
import os
from lxml import etree 
import random

FILEDIR  = "/media/scs4450/hard/umdfaces_batch3"                                             #
FILENAME = "umdfaces_batch3_ultraface.csv" 
IMGSTORE      = "/media/scs4450/hard/luyi/luyi_data/umdface_final/JPEGImages"              #
ANNOTATIONDIR = "/media/scs4450/hard/luyi/luyi_data/umdface_final/Annotations"
SET           = '/media/scs4450/hard/luyi/luyi_data/umdface_final/ImageSets'
SET_MAIN      = '/media/scs4450/hard/luyi/luyi_data/umdface_final/ImageSets/Main'

TRAIN = 'train.txt'
VAL = 'val.txt'
TRAINVAL = 'trainval.txt'
TEST = 'test.txt'
FACES = 'face_distri.txt'
FACES1 = 'face_distri1.txt'
FACES2 = 'face_distri2.txt'

def createXML(trans):      
    annotation = etree.Element("annotation")  

    folder = etree.SubElement(annotation, "folder" )  
    folder.text = trans['folder']
      
    filename = etree.SubElement(annotation, "filename")   
    filename.text = trans['file_name']

    source = etree.SubElement(annotation, "source")   
    source.text = "Unknown"

    owner = etree.SubElement(annotation, "owner")   
    flickrid = etree.SubElement(owner, "flickrid")   
    flickrid.text = "NULL"
    name = etree.SubElement(owner, "name")   
    name.text = "luyi"

    size = etree.SubElement(annotation, "size")   
    width = etree.SubElement(size, "width")   
    width.text = str(trans['img_shape'][1])
    height = etree.SubElement(size, "height")   
    height.text = str(trans['img_shape'][0])
    depth = etree.SubElement(size, "depth")   
    depth.text = str(trans['img_shape'][2])

    segmented = etree.SubElement(annotation, "segmented")   
    segmented.text = "0"

    _object = etree.SubElement(annotation, "object")   
    name = etree.SubElement(_object, "name")   
    name.text = "face"
    pose = etree.SubElement(_object, "pose")   
    pose.text = "Unspecified"
    truncated = etree.SubElement(_object, "truncated")   
    truncated.text = "0"
    difficult = etree.SubElement(_object, "difficult")   
    difficult.text = "0"
    bndbox = etree.SubElement(_object, "bndbox")
    xmin = etree.SubElement(bndbox, "xmin")   
    xmin.text = str(trans['box'][0])
    ymin = etree.SubElement(bndbox, "ymin")   
    ymin.text = str(trans['box'][1])
    xmax = etree.SubElement(bndbox, "xmax")   
    xmax.text = str(int(trans['box'][0]) + int(float(trans['box'][2])))
    ymax = etree.SubElement(bndbox, "ymax")   
    ymax.text = str(int(trans['box'][1]) + int(float(trans['box'][3])))
        
    tree = etree.ElementTree(annotation) 
    save_path = os.path.join(ANNOTATIONDIR, trans['file_name'].split('.')[0]+'.xml')
    tree.write(save_path, pretty_print=True, xml_declaration=True, encoding='utf-8')

def border_img(real_path, send_infos, count):
    fd = open(os.path.join(SET, FACES2), 'a')
    BLACK = [0, 0, 0]
    img = cv2.imread(real_path)
    shape = img.shape
    if shape[0] <= 384 and shape[1] <= 384:
        h_diff = 384 - shape[0]; w_diff = 384 - shape[1]
        constant = cv2.copyMakeBorder(img, int(h_diff/2),int(h_diff-h_diff/2), int(w_diff/2), int(w_diff-w_diff/2), cv2.BORDER_CONSTANT, value=BLACK)
        send_infos['box'][1] = int(float(send_infos['box'][1]))+int(h_diff/2); send_infos['box'][0] = int(float(send_infos['box'][0]))+int(w_diff/2)
    else:
        diff = abs(shape[0]-shape[1])
        if shape[0] < shape[1]:
            constant = cv2.copyMakeBorder(img, int(diff/2),int(diff-diff/2),0,0, cv2.BORDER_CONSTANT, value=BLACK)
            send_infos['box'][0] = int(float(send_infos['box'][0]))
            send_infos['box'][1] = int(float(send_infos['box'][1]))+int(diff/2)
        elif shape[0] > shape[1]:
            constant = cv2.copyMakeBorder(img, 0,0,int(diff/2),int(diff-diff/2), cv2.BORDER_CONSTANT, value=BLACK)
            send_infos['box'][0] = int(float(send_infos['box'][0]))+int(diff/2)
            send_infos['box'][1] = int(float(send_infos['box'][1]))
        else:
            constant = img
            send_infos['box'][0] = int(float(send_infos['box'][0]))
            send_infos['box'][1] = int(float(send_infos['box'][1]))
    side = constant.shape[0]; reshape_img = cv2.resize(constant, (384, 384))
    send_infos['box'][0] = int(int(send_infos['box'][0])/float(side)*384);   send_infos['box'][1] = int(int(send_infos['box'][1])/float(side)*384)
    send_infos['box'][2] = int(float(send_infos['box'][2])/float(side)*384); send_infos['box'][3] = int(float(send_infos['box'][3])/float(side)*384)
    send_infos['img_shape'] = reshape_img.shape; send_infos['file_name'] = str(count)+'.jpg'
    cv2.imwrite(os.path.join(IMGSTORE, send_infos['file_name']), reshape_img)
    fd.write(str(send_infos['box'][2])+' '+str(send_infos['box'][3])+'\n')
    fd.close()

def resize_face(real_path, send_infos, count):
    fd = open(os.path.join(SET, FACES2), 'a')                                             #
    BLACK = [0, 0, 0]
    img = cv2.imread(real_path)
    ratio = (60+(random.randint(-10, 10))) / float(send_infos['box'][3])
    send_infos['box'][0] = str(float(send_infos['box'][0]) * ratio); send_infos['box'][1] = str(float(send_infos['box'][1]) * ratio)
    send_infos['box'][2] = str(float(send_infos['box'][2]) * ratio); send_infos['box'][3] = str(float(send_infos['box'][3]) * ratio)
    img = cv2.resize(img, (int(img.shape[1]*ratio), int(img.shape[0]*ratio)), interpolation=cv2.INTER_AREA)
    shape = img.shape
    if shape[0] <= 384 and shape[1] <= 384:
        h_diff = 384 - shape[0]; w_diff = 384 - shape[1]
        constant = cv2.copyMakeBorder(img, int(h_diff/2),int(h_diff-h_diff/2), int(w_diff/2), int(w_diff-w_diff/2), cv2.BORDER_CONSTANT, value=BLACK)
        send_infos['box'][1] = int(float(send_infos['box'][1]))+int(h_diff/2); send_infos['box'][0] = int(float(send_infos['box'][0]))+int(w_diff/2)
    else:
        diff = abs(shape[0]-shape[1])
        if shape[0] < shape[1]:
            constant = cv2.copyMakeBorder(img, int(diff/2),int(diff-diff/2),0,0, cv2.BORDER_CONSTANT, value=BLACK)
            send_infos['box'][0] = int(float(send_infos['box'][0]))
            send_infos['box'][1] = int(float(send_infos['box'][1]))+int(diff/2)
        elif shape[0] > shape[1]:
            constant = cv2.copyMakeBorder(img, 0,0,int(diff/2),int(diff-diff/2), cv2.BORDER_CONSTANT, value=BLACK)
            send_infos['box'][0] = int(float(send_infos['box'][0]))+int(diff/2)
            send_infos['box'][1] = int(float(send_infos['box'][1]))
        else:
            constant = img
            send_infos['box'][0] = int(float(send_infos['box'][0]))
            send_infos['box'][1] = int(float(send_infos['box'][1]))
    if constant.shape[0] > 384:
        #print type(send_infos['box'][0]), type(send_infos['box'][1])
        center_x = int(send_infos['box'][0]) + int(float(send_infos['box'][2])) / 2; center_y = int(send_infos['box'][1]) + int(float(send_infos['box'][3])) / 2
        x0 = 0; y0 = 0; x1 = 0; y1 = 0
        if center_x - 192 >= 0 and center_x + 192 < constant.shape[1]:
            x0 = center_x - 192; x1 = center_x + 192; send_infos['box'][0] = int(send_infos['box'][0]) - x0
        elif center_x - 192 < 0 and center_x + 192 < constant.shape[1]:
            x0 = 0; x1 = 384
        elif center_x - 192 >= 0 and center_x + 192 >= constant.shape[1]:
            x1 = constant.shape[1]; x0 = x1 - 384; send_infos['box'][0] = int(send_infos['box'][0]) - x0
        else:
            x0 = 0; x1 = constant.shape[1]

        if center_y - 192 >= 0 and center_y + 192 < constant.shape[0]:
            y0 = center_y - 192; y1 = center_y + 192; send_infos['box'][1] = int(send_infos['box'][1]) - y0
        elif center_y - 192 < 0 and center_y + 192 < constant.shape[0]:
            y0 = 0; y1 = 384
        elif center_y - 192 >= 0 and center_y + 192 >= constant.shape[0]:
            y1 = constant.shape[0]; y0 = y1 - 384; send_infos['box'][1] = int(send_infos['box'][1]) - y0
        else:
            y0 = 0; y1 = constant.shape[0]
        constant = constant[y0:y1, x0:x1]
    send_infos['img_shape'] = constant.shape; send_infos['file_name'] = str(count)+'.jpg'
    cv2.imwrite(os.path.join(IMGSTORE, send_infos['file_name']), constant)
    fd.write(str(send_infos['box'][2])+' '+str(send_infos['box'][3])+'\n')
    fd.close()

def main():
    if not os.path.exists(ANNOTATIONDIR):
        os.mkdir(ANNOTATIONDIR)
    if not os.path.exists(IMGSTORE):
        os.mkdir(IMGSTORE)
    if not os.path.exists(SET):
        os.mkdir(SET)
    csv_content = loadCSVFile(os.path.join(FILEDIR,FILENAME))
    cvs_content_part = csv_content[1:,1:10]    
    count = 3000000                                                      #
    for lines in cvs_content_part:
        send_infos = {}
        file_name = lines[0].replace('\n','')
        send_infos['file_name'] = file_name
        print 'Process '+send_infos['file_name']
        send_infos['box'] = lines[3:7]
        real_path = os.path.join(FILEDIR, send_infos['file_name'])
        #border_img(real_path, send_infos, count)
        resize_face(real_path, send_infos, count)
        send_infos['folder'] = 'umdfaces'
        createXML(send_infos)
        count = count + 1
        
def create_data_list():
    if not os.path.exists(SET_MAIN):
        os.mkdir(SET_MAIN)
    lines = os.listdir(ANNOTATIONDIR)
    random.shuffle(lines)
    #print len(lines)
    f_t = open(os.path.join(SET_MAIN, TRAIN), 'w')
    f_v = open(os.path.join(SET_MAIN, VAL), 'w')
    f_tv = open(os.path.join(SET_MAIN, TRAINVAL), 'w')
    f_test = open(os.path.join(SET_MAIN, TEST), 'w')
    count = 1
    for line in lines:
        if count <= 2400:
            f_test.write(line.split('.')[0]+'\n')
        else:
            if count <= 180000:
                f_v.write(line.split('.')[0]+'\n')
                f_tv.write(line.split('.')[0]+'\n')
            else:
                f_t.write(line.split('.')[0]+'\n')
                f_tv.write(line.split('.')[0]+'\n')
        count = count + 1
    f_t.close(); f_v.close(); f_tv.close(); f_test.close()

if __name__ == "__main__":
    #main()
    create_data_list()