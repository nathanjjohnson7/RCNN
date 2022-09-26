from PIL import Image
import numpy as np
import random
import math
import glob
import io
import base64
#import cv2

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

import os
from os import listdir
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from selective_search import selective_search
import time

possible_labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
          'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
          'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

alexnet = T.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)

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
    
    final = intersection/(bb1_size + bb2_size - intersection)
    
    return final

class my_alexnet(nn.Module):
    def __init__(self):
        super(my_alexnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.conv1.weight.data = list(alexnet.children())[0][0].weight.data
        self.conv1.bias.data = list(alexnet.children())[0][0].bias.data
        
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv2.weight.data = list(alexnet.children())[0][3].weight.data
        self.conv2.bias.data = list(alexnet.children())[0][3].bias.data
        
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.conv3 = nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3.weight.data = list(alexnet.children())[0][6].weight.data
        self.conv3.bias.data = list(alexnet.children())[0][6].bias.data
        
        self.conv4 = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4.weight.data = list(alexnet.children())[0][8].weight.data
        self.conv4.bias.data = list(alexnet.children())[0][8].bias.data
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5.weight.data = list(alexnet.children())[0][10].weight.data
        self.conv5.bias.data = list(alexnet.children())[0][10].bias.data
        
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        
        self.linear1 = nn.Linear(in_features=9216, out_features=4096, bias=True)
        self.linear1.weight.data = list(alexnet.children())[2][1].weight.data
        self.linear1.bias.data = list(alexnet.children())[2][1].bias.data
        
        self.linear2 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.linear2.weight.data = list(alexnet.children())[2][4].weight.data
        self.linear2.bias.data = list(alexnet.children())[2][4].bias.data
        
        self.final_layer = nn.Linear(in_features=4096, out_features=21, bias=True)
        T.nn.init.xavier_uniform_(self.final_layer.weight, gain=1.0)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        
    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool3(self.relu(self.conv5(x)))
        x = self.avgpool1(x)
        x = T.flatten(x, start_dim=1)
        x = self.relu(self.linear1(self.dropout(x)))
        x = self.relu(self.linear2(self.dropout(x)))
        x = self.final_layer(x)
        return x

net = my_alexnet()
net.load_state_dict(T.load("rcnn_model.pth"))
net.eval()

img_transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]
        )])

#change accordingly
initial_path = "C:\\my_data\\Pascal_Voc\\VOCdevkit\\VOC2012\\JPEGImages\\"
file_suffix = ".jpg"
specific_img = "2012_001689"
img_path = initial_path + specific_img + file_suffix

def rcnn(img_path):
    img = Image.open(img_path)
    img = np.asarray(img.resize((224, 224)))
    bbs = selective_search(img)
    all_regions = T.zeros(len(bbs), 3, 224, 224)
    for i, bb in enumerate(bbs):
        x_min = int(bb[0])
        y_min = int(bb[1])
        x_max = int(bb[2]) + 1 if int(bb[2]) + 1 < 224 else 223
        y_max = int(bb[3]) + 1 if int(bb[3]) + 1 < 224 else 223        

        region = img[y_min:y_max, x_min:x_max]
        region = np.asarray(Image.fromarray(np.uint8(region)).convert('RGB').resize((224, 224)))
        
        all_regions[i] = img_transform(region)

    output = net(all_regions)
    predicted_labels = T.argmax(output, dim=1)
    probabilities = T.nn.functional.softmax(output, dim=1)
    scores = probabilities[T.arange(output.shape[0]), predicted_labels]

    b = []
    for i, bb in enumerate(bbs):
        if predicted_labels[i].item() != 0:
            b.append([bb, predicted_labels[i].item(), scores[i].item()])
    if len(b) == 0:
        return b
    b.sort(key = lambda x: x[2])

    d = []

    while len(b) != 0:
        highest_scored_bb = b.pop(0)
        d.append(highest_scored_bb)
        bbs_to_del = []
        if len(b) == 0:
            break
        for i, bb_data in enumerate(b):
            if calculate_IoU(highest_scored_bb[0], bb_data[0]) >= 0.3:
                bbs_to_del.append(bb_data)
        for data in bbs_to_del:
            b.remove(data)

    return d

def show_final_predictions(img_path, bb_data, return_img_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    bbox = dict(boxstyle ="round", fc ="0.8")
    img = Image.open(img_path)
    img = np.asarray(img.resize((224, 224)))
    ax.imshow(img)
    for bb, label, _ in bb_data:
        label = possible_labels[label]
        x1, y1, x2, y2 = bb
        bbox = mpatches.Rectangle(
            (x1, y1), (x2-x1), (y2-y1), fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(bbox)
        ax.annotate(label,(x1, y1), xytext =(x1, y1-1))
    #plt.axis('off')
    if return_img_path != None:
        fig.savefig(return_img_path + ".png")
    else:
        plt.show()


if __name__ == "__main__":
    d = rcnn(img_path)
    show_final_predictions(img_path, d, "final_img")
