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
%matplotlib inline

import skimage.io
from selective_search import selective_search
import matplotlib.patches as mpatches
from efficient_graph_img_seg import display_segmentation_ds

#use display_segmentation_ds to display segmentation from efficent graph-based img seg and selective search

def show_selective_search_bounding_boxes(img_path, boxes, seperate = False, new_filename = None):
    img = Image.open(image_path)
    img = np.asarray(img.resize((224, 224)))
    if seperate:
        for x1, y1, x2, y2 in boxes:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img)
            bbox = mpatches.Rectangle(
                (x1, y1), (x2-x1), (y2-y1), fill=False, edgecolor='red', linewidth=5)
            ax.add_patch(bbox)
            #plt.axis('off')
            plt.show()
        return
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        for x1, y1, x2, y2 in boxes:
            bbox = mpatches.Rectangle(
                (x1, y1), (x2-x1), (y2-y1), fill=False, edgecolor='red', linewidth=5)
            ax.add_patch(bbox)
        #plt.axis('off')
        if new_filename != None:
            fig.savefig(new_filename + ".png")
        else:
            plt.show()
