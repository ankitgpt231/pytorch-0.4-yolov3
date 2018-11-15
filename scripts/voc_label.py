import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob


classes = ["surgicaltool"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(annotation, text_path):
    in_file = open(annotation)
    tree=ET.parse(in_file)
    root = tree.getroot()
    file_name, _ = os.path.splitext(root.find('filename').text)
    out_file = open(txt_path+file_name+'.txt', 'w')
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

xml_path = '/home/ankit.gupta/tool_detect_yolo/dataset/train_annot_folder/'
txt_path = '/home/ankit.gupta/yolo_pytorch/pytorch-0.4-yolov3/data/annotations_txt/'
dataset = '/home/ankit.gupta/yolo_pytorch/pytorch-0.4-yolov3/data/'
split_ratio = 0.2

if not os.path.exists(txt_path):
    os.mkdir(txt_path)
annotations_xml = glob.glob(xml_path+'*.xml')
list_file_train = open(dataset + 'train.txt', 'w')
list_file_valid = open(dataset + 'valid.txt', 'w')
sample_factor = int(1/split_ratio)
for i in range(len(annotations_xml)):
    tree=ET.parse(annotations_xml[i])
    root = tree.getroot()
    image_id = root.find('path').text
    if (i+1)%sample_factor == 0:
        list_file_valid.write(image_id+'\n')
    else:
        list_file_train.write(image_id+'\n')
    convert_annotation(annotations_xml[i], txt_path)
