from PIL import Image
import numpy as np
import random
import math
import glob
import io
import base64
import os

from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

from operator import itemgetter

import matplotlib
import matplotlib.pyplot as plt

def make_graph():
    img_height = 224
    img_width = 224
    numbers = np.arange(img_height*img_width)
    numbers = numbers.reshape((img_height, img_width))
    neighbours_i = [-1, -1, -1, 0, 0, 1, 1, 1]
    neighbours_j = [-1, 0, 1, -1, 1, -1, 0, 1]
    graph = {}
    edges = []
    edges = set()
    for i in range(img_height):
        for j in range(img_width):
            possible_neighbours = []
            for k in range(8):
                if(i+neighbours_i[k] >=0 and i+neighbours_i[k]<=img_height-1 and j+neighbours_j[k] >=0 and j+neighbours_j[k]<=img_width-1):
                    possible_neighbours.append(numbers[i+neighbours_i[k]][j+neighbours_j[k]])
            graph[numbers[i][j]] = possible_neighbours
            for l in possible_neighbours:
                #edges.append([numbers[i][j], l, 0])
                if numbers[i][j] <= l:
                    edges.add((numbers[i][j], l, 0))
                else:
                    edges.add((l, numbers[i][j], 0))
    return graph, [list(edge) for edge in edges]

def index_to_indices(index, height = 224, width = 224):
    x = math.floor(index/width)
    y = index - (x*width)
    return x,y

def add_weight_to_edges_rgb(edges, img):
    for edge in edges:
        x1, y1 = index_to_indices(edge[0])
        vertex1 = np.array([x1, y1, int(img[x1, y1, 0]), int(img[x1, y1, 1]), int(img[x1, y1, 2])])
        x2, y2 = index_to_indices(edge[1])
        vertex2 = np.array([x2, y2, int(img[x2, y2, 0]), int(img[x2, y2, 1]), int(img[x2, y2, 2])])
        edge[-1] = (vertex1[2]-vertex2[2])**2 + (vertex1[3]-vertex2[3])**2 + (vertex1[4]-vertex2[4])**2 
    return edges

def add_weight_to_edges(edges, img_channel):
    for edge in edges:
        edge[-1] = abs(int(img_channel[index_to_indices(edge[0])]) - int(img_channel[index_to_indices(edge[1])]))
    return edges


def display_segmentation_ds(ds):
    my_image = np.zeros((224, 224, 3))
    parents = {}
    for i in range(len(ds.parent)):
        parent = ds.find_set(i)
        if parent in parents:
            parents[parent].append(i)
        else:
            parents[parent] = [i]
    for parent in parents.keys():
        component = parents[parent]
        color = np.array([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
        for vertex in component:
            #print(type(vertex))
            indicies = index_to_indices(int(vertex))
            my_image[indicies] = color
    return parents, my_image

class DisjointSet:
    def __init__(self, num):
        self.parent = [i for i in range(num)]
        self.rank = [0 for i in range(num)]
        self.int = [0 for i in range(num)]
        self.size = [1 for i in range(num)]
    def find_set(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find_set(self.parent[x])
        return self.parent[x]
    def link(self, x, y, edge_weight):
        if self.rank[x] > self.rank[y]:
            self.parent[y] = x
            self.int[x] = edge_weight
            self.size[x] += self.size[y]
        else:
            self.parent[x] = y
            self.int[y] = edge_weight
            self.size[y] += self.size[x]
            if self.rank[x] == self.rank[y]:
                self.rank[y] +=1
    def union(self, x, y, edge_weight):
        self.link(self.find_set(x), self.find_set(y), edge_weight)
    def get_int(self, x):
        return self.int[x]
    def get_size(self, x):
        return self.size[x]

def eff_graph_image_seg(img, k=300, sigma=0.8, rgb=False, show_img=True):
    #currently only works on one channel
    img = gaussian_filter(img, sigma=sigma)
    graph, edges = make_graph()
    #red channel is being used
    if rgb:
        edges = add_weight_to_edges_rgb(edges, img)
    else:
        edges = add_weight_to_edges(edges, img)
    edges = sorted(edges, key=itemgetter(2))
    
    ds = DisjointSet(224*224)
    
    for i, edge in enumerate(edges):
        vertex1 = edge[0]
        vertex2 = edge[1]
        
        pos1 = ds.find_set(vertex1)
        pos2 = ds.find_set(vertex2)
        if pos1 == pos2:
            continue
        if edge[2] <= min(ds.get_int(pos1) + (k/ds.get_size(pos1)), ds.get_int(pos2) + (k/ds.get_size(pos2))):
            ds.link(pos1, pos2, edge[2])
    
    segments, image = display_segmentation_ds(ds)
    if show_img:
        print()
        print()
        print("******************************************")
        plt.imshow(image/255, interpolation='nearest')
        plt.show()
    return graph, edges, segments, ds
