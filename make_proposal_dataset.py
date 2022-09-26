import xml.etree.ElementTree as ET
import os
from os import listdir
from os.path import join

from PIL import Image
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from selective_search import selective_search
import time

image_folder_path = "C:\\my_data\\Pascal_Voc\\VOCdevkit\\VOC2012\\JPEGImages"
annotation_file_path = "C:\\my_data\\Pascal_Voc\\VOCdevkit\\VOC2012\\Annotations"
annotation_files = listdir(annotation_file_path)
num_images = len(annotation_files)
new_files = [file for file in annotation_files if file[:4] != "2007" and file[:4] != "2008"]
num_images = len(new_files)

def get_img(name):
    image_path = os.path.join(image_folder_path, name)
    img = Image.open(image_path)
    img = img.resize((224, 224))
    return np.asarray(img)

def calculate_IoU(bb1, bb2):
    bb1_size = (bb1[2]-bb1[0])*(bb1[3]-bb1[1])
    bb2_size = (bb2[2]-bb2[0])*(bb2[3]-bb2[1])
    
    Xs = [[1, bb1[0]], [1, bb1[2]], [2, bb2[0]], [2, bb2[2]]]
    Ys = [[1, bb1[1]], [1, bb1[3]], [2, bb2[1]], [2, bb2[3]]]
    
    Xs.sort(key = lambda x: x[1])
    Ys.sort(key = lambda x: x[1])

    if Xs[0][0] == Xs[1][0] or Ys[0][0] == Ys[1][0]:
        return 0
    
    x_overlap = Xs[2][1] - Xs[1][1]
    y_overlap = Ys[2][1] - Ys[1][1]

    intersection = x_overlap * y_overlap
    
    return intersection/(bb1_size + bb2_size - intersection)

for current_img_num, file in enumerate(new_files):
    start = time.time()
    tree = ET.parse(join(annotation_file_path, file))
    root = tree.getroot()
    filename = root.find("filename").text
    size = root.find("size")
    height = int(size.find("height").text)
    width = int(size.find("width").text)
    x_scale = width/224
    y_scale = height/224
    dims = [height, width]
    names = []
    bbs = []
    for item in root.findall("object"):
        name = item.find("name").text
        names.append(name)
        bndbox = item.find("bndbox")
        bb = []
        bb.append(round(float(bndbox.find("xmin").text)/x_scale))
        bb.append(round(float(bndbox.find("ymin").text)/y_scale))
        bb.append(round(float(bndbox.find("xmax").text)/x_scale))
        bb.append(round(float(bndbox.find("ymax").text)/y_scale))
        bbs.append(bb)
    img = get_img(filename)
    predicted_bounding_boxes = selective_search(img)
    for pred_BB in predicted_bounding_boxes:
        max_IoU = 0
        max_label = "background"
        for i, label_BB in enumerate(bbs):
            IoU = calculate_IoU(pred_BB, label_BB)
            if IoU > max_IoU:
                max_IoU = IoU
                max_label = names[i]
        with open("real_dataset.txt","a+") as f:
            f.write("{} {} {} {} {} {} {}\n".format(filename, pred_BB[0], pred_BB[1], pred_BB[2], pred_BB[3], max_label, max_IoU))
    end = time.time()
    print(current_img_num, "/", len(new_files), "        ", end - start, "seconds")
