import torchmetrics
from PIL import Image
#import skimage.transform
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

import xml.etree.ElementTree as ET
import os
from os import listdir
from os.path import join

from PIL import Image
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
%matplotlib inline

from selective_search import selective_search
import time

if T.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"  

def one_hot_encode_label(label, num_categories):
    final = T.zeros(label.shape[0], num_categories)
    new_label = T.cat((T.arange(label.shape[0]).unsqueeze(0), label.unsqueeze(0)))
    final[new_label[0], new_label[1]] = 1
    return final

alexnet = T.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True).to(device)

possible_labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
          'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
          'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class rcnn_dataset(Dataset):
    def __init__(self, info_list, image_folder_path):        
        self.info = info_list
        self.image_folder_path = image_folder_path
        
    def __len__(self):
        return len(self.info)
    
    def __getitem__(self, index):
        if T.is_tensor(index):
            idx = idx.tolist()
        data = self.info[index]
        image_path = os.path.join(self.image_folder_path, data[0])
        img = Image.open(image_path)
        img = np.asarray(img.resize((224, 224)))
        
        x_min = int(data[1])
        y_min = int(data[2])
        x_max = int(data[3]) + 1 if int(data[3]) + 1 < 224 else 223
        y_max = int(data[4]) + 1 if int(data[4]) + 1 < 224 else 223        
    
        region = img[y_min:y_max, x_min:x_max]
        region = np.asarray(Image.fromarray(np.uint8(region)).convert('RGB').resize((224, 224)))
        #plt.imshow(region, interpolation = "nearest")
        #plt.show()
        
        img_transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]
        )])
       
        region = img_transform(region)
        
        label = possible_labels.index(data[5])

        return region, label
train_test_split = 0.8
image_folder_path = "C:\\my_data\\Pascal_Voc\\VOCdevkit\\VOC2012\\JPEGImages"
background_bb_path = "C:\\my_code\\AI\\efficent_graph_based_img_seg\\real_dataset.txt"
ground_truth_path = "C:\\my_code\\AI\\efficent_graph_based_img_seg\\ground_truths.txt"
with open(background_bb_path) as f:
    background_bb_data = [string.split() for string in f.readlines()]
with open(ground_truth_path) as f:
    positives = [string.split() for string in f.readlines()]
    
negatives = []

for line in background_bb_data:  
    if float(line[6]) <= 0.3:
        changed_line = line[:6]
        changed_line[-1] = 'background'
        negatives.append(changed_line)
        
positive_split = round(len(positives)*train_test_split)
negative_split = round(len(negatives)*train_test_split) 

positive_train = positives[:positive_split]
positive_test = positives[positive_split:]

negative_train = negatives[:negative_split]
negative_test = negatives[negative_split:]

positive_train_dataset = rcnn_dataset(positive_train, image_folder_path)
positive_test_dataset = rcnn_dataset(positive_test, image_folder_path)

negative_train_dataset = rcnn_dataset(negative_train, image_folder_path)
negative_test_dataset = rcnn_dataset(negative_test, image_folder_path)

p_train_dataloader = DataLoader(positive_train_dataset, batch_size=32, shuffle=True, num_workers=0)
n_train_dataloader = DataLoader(negative_train_dataset, batch_size=96, shuffle=True, num_workers=0)

p_test_dataloader = DataLoader(positive_test_dataset, batch_size=32, shuffle=True, num_workers=0)
n_test_dataloader = DataLoader(negative_test_dataset, batch_size=96, shuffle=True, num_workers=0)

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
    
net = my_alexnet().to(device)
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)
epochs = 60000 #200

def train(data, label, criterion, optimizer):
    net.train()
    output = net(data.to(device))
    one_hot_label = one_hot_encode_label(label, 21)
    output_loss = criterion(output.to(device), one_hot_label.to(device)).to(device)
    optimizer.zero_grad()
    output_loss.backward()
    optimizer.step()
    
    final_output = T.argmax(output, dim=1)
    precision, recall = torchmetrics.functional.precision_recall(output, label, average="macro", num_classes = 21)
    
    return output, final_output, output_loss, precision.item(), recall.item()

def val(data, label, criterion, optimizer):
    net.eval()
    output = net(data.to(device))
    one_hot_label = one_hot_encode_label(label, 21)
    output_loss = criterion(output.to(device), one_hot_label.to(device)).to(device)
    
    final_output = T.argmax(output, dim=1)
    precision, recall = torchmetrics.functional.precision_recall(output, label, average="macro", num_classes = 21)
    
    return output, final_output, output_loss, precision.item(), recall.item()

train_precisions = []
train_recalls = []
train_losses = []

val_precisions = []
val_recalls = []
val_losses = []

def train_loop(epochs):
    p_train_gen = iter(p_train_dataloader)
    n_train_gen = iter(n_train_dataloader)
    
    p_test_gen = iter(p_test_dataloader)
    n_test_gen = iter(n_test_dataloader)
    
    for i in range(epochs):
        try:
            p_train_data, p_train_label = next(p_train_gen)
        except StopIteration:
            p_train_gen = iter(p_train_dataloader)
            p_train_data, p_train_label = next(p_train_gen)
        try:
            n_train_data, n_train_label = next(n_train_gen)
        except StopIteration:
            n_train_gen = iter(n_train_dataloader)
            n_train_data, n_train_label = next(n_train_gen)
        try:
            p_test_data, p_test_label = next(p_test_gen)
        except StopIteration:
            p_test_gen = iter(p_test_dataloader)
            p_test_data, p_test_label = next(p_test_gen)
        try:
            n_test_data, n_test_label = next(n_test_gen)
        except StopIteration:
            n_test_gen = iter(n_test_dataloader)
            n_test_data, n_test_label = next(n_test_gen)
        
        train_data = T.cat((p_train_data, n_train_data))
        train_labels = T.cat((p_train_label, n_train_label))
        
        indices = T.randperm(train_data.size()[0])
        train_data = train_data[indices]
        train_labels = train_labels[indices]
        
        test_data = T.cat((p_test_data, n_test_data))
        test_labels = T.cat((p_test_label, n_test_label))
        
        indices2 = T.randperm(test_data.size()[0])
        test_data = test_data[indices2]
        test_labels = test_labels[indices2]
        
        train_output, train_final_output, train_loss, train_precision, train_recall = train(train_data, train_labels, loss, optimizer)
        train_losses.append(train_loss.item())
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        
        test_output, test_final_output, test_loss, test_precision, test_recall = val(test_data, test_labels, loss, optimizer)
        val_losses.append(test_loss.item())
        val_precisions.append(test_precision)
        val_recalls.append(test_recall)
        
        print()
        print("Epoch {}".format(i))
        print("Train Loss: {}, train_precision: {}, train_recall {}".format(train_loss.item(), train_precision, train_recall))
        print("Val Loss: {}, val_precision: {}, val_recall {}".format(test_loss.item(), test_precision, test_recall))
        if(test_loss.item() <= min(val_losses)):
            print("Saved model switch.")
            T.save(net.state_dict(), "C:\\my_code\\AI\\efficent_graph_based_img_seg\\rcnn_model.pth")
        print()
        
        with open("rcnn_log.txt","a+") as f:
            f.write("\n")
            f.write("Epoch {}\n".format(i))
            f.write("Train Loss: {}, train_precision: {}, train_recall {}".format(train_loss.item(), train_precision, train_recall))
            f.write("Val Loss: {}, val_precision: {}, val_recall {}".format(test_loss.item(), test_precision, test_recall))
            if(test_loss.item() <= min(val_losses)):
                f.write("Saved model switch.\n")
                T.save(net.state_dict(), "C:\\my_code\\AI\\efficent_graph_based_img_seg\\rcnn_model.pth")
            f.write("\n")
            
train_loop(epochs)
