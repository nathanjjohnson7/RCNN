from PIL import Image
import numpy as np
import random
import math
import glob
import io
import base64
import os

import time

from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

from operator import itemgetter

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from efficient_graph_img_seg import display_segmentation_ds
from efficient_graph_img_seg import eff_graph_image_seg

import torch as T
import torch.nn as nn

sobel_edge_detectors = np.array([[[-1,  0,  1],
                                  [-2,  0,  2],
                                  [-1,  0,  1]],

                                 [[ 1,  0, -1],
                                  [ 2,  0, -2],
                                  [ 1,  0, -1]],

                                 [[ 1,  2,  1],
                                  [ 0,  0,  0],
                                  [-1, -2, -1]],

                                 [[-1, -2, -1],
                                  [ 0,  0,  0],
                                  [ 1,  2,  1]],

                                 [[ 0,  1,  2],
                                  [-1,  0,  1],
                                  [-2, -1,  0]],

                                 [[ 0, -1, -2],
                                  [ 1,  0, -1],
                                  [ 2,  1,  0]],

                                 [[-2, -1,  0],
                                  [-1,  0,  1],
                                  [ 0,  1,  2]],

                                 [[ 2,  1,  0],
                                  [ 1,  0, -1],
                                  [ 0, -1, -2]]])

def index_to_indices(index, height = 224, width = 224):
    x = math.floor(index/width)
    y = index - (x*width)
    return x,y

def make_bins(num_bins, min_value, max_value):
    bin_range = (max_value-min_value)/num_bins
    return [min_value+(i*bin_range) for i in range(num_bins+1)]

def histogram_intersection(hist1, hist2):
    combined = np.concatenate((np.expand_dims(hist1, axis=0), np.expand_dims(hist2, axis=0)), axis=0)
    mins = np.min(combined, axis=0)
    return mins.sum()

def histogram_propagation(region1_size, region2_size, hist1, hist2):
    return (region1_size*hist1 + region2_size*hist2)/(region1_size + region2_size)

def image_gradients(img):
    img = gaussian_filter(img, sigma=1)
    sobel = T.from_numpy(sobel_edge_detectors).unsqueeze(1).float()
    conv = nn.Conv2d(1, 8, (3,3), padding = 1, padding_mode = "replicate")
    conv.weight.data = sobel
    
    red_derivatives = conv(T.from_numpy(img[:,:,0]).unsqueeze(0).unsqueeze(0).float())
    green_derivatives = conv(T.from_numpy(img[:,:,1]).unsqueeze(0).unsqueeze(0).float())
    blue_derivatvies = conv(T.from_numpy(img[:,:,2]).unsqueeze(0).unsqueeze(0).float())
    
    derivative = T.cat((red_derivatives, green_derivatives, blue_derivatvies), 0)
    return derivative

def make_texture_histogram(Xs, Ys, img, img_derivative):
    current_histogram = np.array([])
    for i in range(img_derivative.shape[0]):
        for j in range(img_derivative.shape[1]):
            hist = np.histogram(img_derivative[i, j, Ys, Xs].detach().numpy(), bins=10)
            current_histogram = np.concatenate((current_histogram, hist[0]))
    return current_histogram/current_histogram.sum()

def make_color_histogram(Xs, Ys, img):
    bins = make_bins(25, 0, 256)
    red_histogram = np.histogram(img[Ys,Xs,0], bins=bins)[0]
    green_histogram = np.histogram(img[Ys,Xs,1], bins=bins)[0] 
    blue_histogram = np.histogram(img[Ys,Xs,2], bins=bins)[0]
    final = np.concatenate((red_histogram, green_histogram, blue_histogram))
    final = final/final.sum()
    return final

def make_histogram(segments, img):
    derivative = image_gradients(img)
    color_dict = {}
    texture_dict = {}
    BB_points_dict = {}
    all_bounding_boxes = []
    for segment in segments:
        Xs = []
        Ys = []
        for value in segments[segment]:
            x, y = index_to_indices(value)
            #x and y are switched becuase of the way indexing works
            Xs.append(y)
            Ys.append(x)
        color_hist = make_color_histogram(Xs, Ys, img)
        texture_hist = make_texture_histogram(Xs, Ys, img, derivative)
        color_dict[segment] = color_hist
        texture_dict[segment] = texture_hist
        BB_points_dict[segment] = [[min(Xs), min(Ys)], [max(Xs), max(Ys)]]
        all_bounding_boxes.append([min(Xs), min(Ys), max(Xs), max(Ys)])
    return color_dict, texture_dict, BB_points_dict, all_bounding_boxes

def get_joint_region_bb(region1, region2, BB_points):
    coords1 = BB_points[region1][:]
    coords2 = BB_points[region2][:]
    coords1.extend(coords2)
    coords = np.asarray(coords1)
    Xs = coords[:, 0]
    Ys = coords[:, 1]
    bb_coordinates = [[Xs.min(), Ys.min()], [Xs.max(), Ys.max()]]
    size = (Xs.max() - Xs.min() + 1) * (Ys.max() - Ys.min() + 1)
    return bb_coordinates, size

def calculate_similarity(region1, region2, image_size, segments, color_histograms, texture_histograms, BB_points):
    s_color = histogram_intersection(color_histograms[region1], color_histograms[region2])
    s_texture = histogram_intersection(texture_histograms[region1], texture_histograms[region2])
    s_size = 1 - ((len(segments[region1]) + len(segments[region2]))/image_size)
    joint_bb_coords, joint_bb_size = get_joint_region_bb(region1, region2, BB_points)
    s_fill = 1 - ((joint_bb_size - len(segments[region1]) - len(segments[region2]))/image_size)
    return s_color + s_size + s_fill  #s_color + s_texture + s_size + s_fill

def process_linking_overhead(region1, region2, new_region, segments, color_histograms, texture_histograms, BB_points, all_BBs):
    new_color_histogram = histogram_propagation(len(segments[region1]), len(segments[region2]), color_histograms[region1], color_histograms[region2])
    new_texture_histogram = histogram_propagation(len(segments[region1]), len(segments[region2]), texture_histograms[region1], texture_histograms[region2])
    new_bb_coordinates, _ = get_joint_region_bb(region1, region2, BB_points)
    
    del color_histograms[region1]
    del color_histograms[region2]
    del texture_histograms[region1]
    del texture_histograms[region2]
    del BB_points[region1]
    del BB_points[region2]
    
    color_histograms[new_region] = new_color_histogram
    texture_histograms[new_region] = new_texture_histogram
    BB_points[new_region] = new_bb_coordinates
    all_BBs.append([new_bb_coordinates[0][0], new_bb_coordinates[0][1], new_bb_coordinates[1][0], new_bb_coordinates[1][1]])

def selective_search(img, seg_k=800, show_images = False):
    image_size = img.shape[0]*img.shape[1]
    #image segmentation using blue channel of image
    graph, edges, segments, ds = eff_graph_image_seg(img[:, :, 1], k=seg_k, show_img=False)
    color_histograms, texture_histograms, BB_points, all_BBs = make_histogram(segments, img)
    similarity_sets = {}
    neighbours = {key: [] for key in segments.keys()}

    for edge in edges:
        parent1 = ds.find_set(edge[0])
        parent2 = ds.find_set(edge[1])
        if parent1 != parent2:
            if parent1<parent2:
                new_pair = (parent1, parent2)
            else:
                new_pair = (parent2, parent1)
            if new_pair not in similarity_sets:
                neighbours[parent1].append(parent2)
                neighbours[parent2].append(parent1)
                similarity = calculate_similarity(new_pair[0], new_pair[1], image_size, segments, color_histograms, texture_histograms, BB_points)
                similarity_sets[new_pair] = similarity
                
    while len(similarity_sets) > 0:
        most_similar = max(similarity_sets.items(), key=itemgetter(1))[0]
        ds.link(most_similar[0], most_similar[1], 0)
        keys_to_delete = []
        for key in similarity_sets.keys():
            if key[0] == most_similar[0] or key[1] == most_similar[0] or key[0] == most_similar[1] or key[1] == most_similar[1]:
                keys_to_delete.append(key)
        for key in keys_to_delete:
              del similarity_sets[key]
        segment1_neighbours = neighbours[most_similar[0]][:]
        segment1_neighbours.remove(most_similar[1])
        segment2_neighbours = neighbours[most_similar[1]][:]
        segment2_neighbours.remove(most_similar[0])
        new_neighbours = segment1_neighbours + segment2_neighbours
        #remove duplicates
        new_neighbours = list(dict.fromkeys(new_neighbours))
        del neighbours[most_similar[0]]
        del neighbours[most_similar[1]]
        new_region = ds.find_set(most_similar[0])
        process_linking_overhead(most_similar[0], most_similar[1], new_region, segments, color_histograms, texture_histograms, BB_points, all_BBs)
        assert new_region == ds.find_set(most_similar[1])
        neighbours[new_region] = new_neighbours
        for neighbour in neighbours[new_region]:
            if most_similar[0] in neighbours[neighbour]:
                neighbours[neighbour].remove(most_similar[0])
            if most_similar[1] in neighbours[neighbour]:
                neighbours[neighbour].remove(most_similar[1])
            neighbours[neighbour].append(new_region)
            similarity = calculate_similarity(neighbour, new_region, image_size, segments, color_histograms, texture_histograms, BB_points)
            if neighbour<new_region:
                new_pair = (neighbour, new_region)
            else:
                new_pair = (new_region, neighbour)
            similarity_sets[new_pair] = similarity
        new_segment = segments[most_similar[0]][:] + segments[most_similar[1]][:]
        del segments[most_similar[0]]
        del segments[most_similar[1]]
        segments[new_region] = new_segment
        
        if show_images:
            _, image = display_segmentation_ds(ds)
            x1 = BB_points[new_region][0][0]
            y1 = BB_points[new_region][0][1]
            x2 = BB_points[new_region][1][0]
            y2 = BB_points[new_region][1][1]
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image/255)
            bbox = mpatches.Rectangle(
                (x1, y1), (x2-x1), (y2-y1), fill=False, edgecolor='red', linewidth=5)
            ax.add_patch(bbox)
            plt.show()
            
    set_BBs = {tuple(s) for s in all_BBs}
    BBs_with_low_area = [bb for bb in set_BBs if (bb[2]-bb[0])*(bb[3]-bb[1]) < 40]
    for BB in BBs_with_low_area:
        set_BBs.remove(BB)
    return set_BBs
