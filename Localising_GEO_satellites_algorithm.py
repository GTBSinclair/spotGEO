#!/usr/bin/env python
# coding: utf-8

# In[1]:


#####
# Script: spotGEO starter kit
# Modifications: Giovanni Tognini Bonelli Sinclair
# Date: 07/12/21
#####


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
from collections import defaultdict
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import itertools
import random
from skimage import measure
from scipy.signal import convolve2d
import math as m


# ## Reading a frame from the SpotGEO dataset

# In[2]:


path = 'C:/Users/sincl/OneDrive/Desktop/Cranfield University/spotGEO project/Dataset/train/1/5.png'
def read_image(path):
    return plt.imread(path)
def plot_img(path):
    example_image=read_image(path)
#     print(example_image.shape)
#     plt.imshow(example_image)
#     plt.axis('on')
#     plt.xlim(600,640)
#     plt.ylim(100,150)
#     plt.grid()
#     plt.show()
    return example_image
    
# example_image = plot_img(path)


# In[3]:


example_image=read_image(path)
# print(example_image.shape)
# plt.imshow(example_image)
# plt.axis('on')
# plt.xlim(500,620)
# plt.ylim(100,200)
# plt.grid()
# plt.show()


# ## Reading the annotation file
# 
# ### Format of train_annotation:
# Train_annotation [sequenceID] [frameID]

# In[4]:


path = 'C:/Users/sincl/OneDrive/Desktop/Cranfield University/spotGEO project/Dataset/train_anno.json'
def read_annotation_file(path):
    with open(path) as annotation_file:
        annotation_list = json.load(annotation_file)
    # Transform list of annotations into dictionary
    annotation_dict = {}
    for annotation in annotation_list:
        sequence_id = annotation['sequence_id']
        if sequence_id not in annotation_dict:
            annotation_dict[sequence_id] = {}
        annotation_dict[sequence_id][annotation['frame']] = annotation['object_coords']
    return annotation_dict

train_annotation=read_annotation_file(path)
# print(train_annotation[1][1])
# print(train_annotation[1][5])


# # Generating the dataset

# In[5]:


random.seed(0)

def random_different_coordinates(coords, size_x, size_y, pad):
    """ Returns a random set of coordinates that is different from the provided coordinates, 
    within the specified bounds.
    The pad parameter avoids coordinates near the bounds."""
    good = False
    while not good:
        good = True
        c1 = random.randint(pad + 1, size_x - (pad + 1))
        c2 = random.randint(pad + 1, size_y -( pad + 1))
        for c in coords:
            if c1 == c[0] and c2 == c[1]:
                good = False
                break
    return (c1,c2)

def extract_neighborhood(x, y, arr, radius):
    """ Returns a 1-d array of the values within a radius of the x,y coordinates given """
    return arr[(x - radius) : (x + radius + 1), (y - radius) : (y + radius + 1)].ravel()

def check_coordinate_validity(x, y, size_x, size_y, pad):
    """ Check if a coordinate is not too close to the image edge """
    return x >= pad and y >= pad and x + pad < size_x and y + pad < size_y

def generate_labeled_data(image_path, annotation, nb_false, radius):
    """ For one frame and one annotation array, returns a list of labels 
    (1 for true object and 0 for false) and the corresponding features as an array.
    nb_false controls the number of false samples
    radius defines the size of the sliding window (e.g. radius of 1 gives a 3x3 window)"""
    features,labels = [],[]
    im_array = read_image(image_path)
    # True samples
    for obj in annotation:
        obj = [int(x + .5) for x in obj] #Project the floating coordinate values onto integer pixel coordinates.
        # For some reason the order of coordinates is inverted in the annotation files
        if check_coordinate_validity(obj[1],obj[0],im_array.shape[0],im_array.shape[1],radius):
            features.append(extract_neighborhood(obj[1],obj[0],im_array,radius))
            labels.append(1)
    # False samples
    for i in range(nb_false):
        c = random_different_coordinates(annotation,im_array.shape[1],im_array.shape[0],radius)
        features.append(extract_neighborhood(c[1],c[0],im_array,radius))
        labels.append(0)
    return np.array(labels),np.stack(features,axis=1)


# In[6]:


def generate_labeled_set(annotation_array, path, sequence_id_list, radius, nb_false):
    # Generate labeled data for a list of sequences in a given path
    labels,features = [],[]
    for seq_id in sequence_id_list:
        for frame_id in range(1,6):
            d = generate_labeled_data(f"{path}{seq_id}/{frame_id}.png",
                                    annotation_array[seq_id][frame_id],
                                    nb_false,
                                    radius)
            labels.append(d[0])
            features.append(d[1])
    return np.concatenate(labels,axis=0), np.transpose(np.concatenate(features,axis=1))


# In[7]:


radius=3
path = 'C:/Users/sincl/OneDrive/Desktop/Cranfield University/spotGEO project/Dataset/train/'
train_labels, train_features = generate_labeled_set(train_annotation,path, range(1,101), radius, 10)
# print(train_labels.shape)
# print(train_labels)
# print(train_features.shape)


# ### True satellite

# In[17]:


# print(train_labels[0])
kernel1 = train_features[0].reshape((7,7))
# plt.imshow(kernel1, cmap='gray',vmin=0,vmax=1)
# plt.axis('off')
# plt.show()
# print(kernel1)


# ### False satellite

# In[18]:


# print(train_labels[5])
false_img = train_features[5].reshape((7,7))
# plt.imshow(false_img, cmap='gray',vmin=0,vmax=1)
# plt.axis('off')
# plt.show()


# In[10]:


# kernel2 = np.array([[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,5,5,5,0,0,0],[0,0,5,5,5,5,0],[0,0,0,5,5,5,0],[0,0,0,0,5,5,5],[0,0,0,0,0,0,0]])
# plt.imshow(kernel2, vmin=0, vmax=1)


# In[11]:


# path = 'C:/Users/sincl/OneDrive/Desktop/Cranfield University/spotGEO project/Dataset/train/1/5.png'
# plt.figure(figsize = (10,10))
# example_image = plot_img(path)


# In[12]:


# for i in range(len(example_image)-8):
#     for j in range(len(example_image[0])-8):
#         if example_image[i][j] > 0.3:
#             try:
# #                 print(max(example_image[i+4][j-4:j+4]))
#                 if max(example_image[i+4][j-4:j+4]) < 0.21 and max(example_image[i-4][j-4:j+4]) < 0.21 and max(example_image[j+4][i-4:i+4]) < 0.21 and max(example_image[j-4][i-4:i+4]) < 0.21:
#                     example_image[i][j] = -1
# #                     print('worked')
#             except ValueError as error:
#                 continue
# #                 print('error')
#             except IndexError as ie:
#                 continue
# #                 print('error')


# In[13]:


gain = 5

for i in range(len(example_image)):
    for j in range(len(example_image[0])):
        example_image[i][j] = m.tanh(gain*example_image[i][j])
#         if example_image[i][j] < 0.2:
#             example_image[i][j] = 0
#         if example_image[i][j] > 0.29:
#             example_image[i][j] = max(example_image[i])
            
            # Try: output = tanh(gain * input)


# In[14]:


# plt.figure(figsize = (10,10))
# plt.imshow(example_image)
# plt.grid()


# ## Testing the convolve2D method

# In[15]:


# for i in range(len(kernel)):
#     for j in range(len(kernel)):
#         kernel[i][j] = kernel[i][j]*150
        
# for i in range(len(example_image)):
#     for j in range(len(example_image[0])):
#         example_image[i][j] = example_image[i][j]*150


# In[16]:


# print(example_image[340][70:110])


# In[ ]:




