#!/usr/bin/env python
# coding: utf-8

# In[1]:


#####
# Script: Giovanni Tognini Bonelli Sinclair
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

from Localising_GEO_satellites_algorithm import read_annotation_file,read_image, plot_img,random_different_coordinates,extract_neighborhood,check_coordinate_validity, generate_labeled_data,generate_labeled_set;
from Main_algorithm_v2 import Brightness_analysis # ***** Main_algorithm_v2.py needs to be in the same directory as this script for this to work


# In[2]:


path1 = 'C:/Users/sincl/OneDrive/Desktop/Cranfield University/spotGEO project/Dataset/train_anno.json'
path2 = 'C:/Users/sincl/OneDrive/Desktop/Cranfield University/spotGEO project/Dataset/train/'
path3 = 'C:/Users/sincl/OneDrive/Desktop/Cranfield University/spotGEO project/Dataset/train/1/5.png'

radius=3
train_annotation=read_annotation_file(path1)

labels, features = generate_labeled_set(train_annotation,path2, range(101,103), radius, 500)

print(labels.shape)
print(labels)
print(features.shape)

ID = 0
ID2 = 3


# In[3]:


# threshold = np.arange(0.001,0.015,0.001)

# for i in range(len(threshold)):
#     results = Brightness_analysis.classify(features,labels,ID,ID2,threshold[i])
#     print("Kappa =",cohen_kappa_score(results[0],labels))


# In[4]:


threshold = 0.012
results = Brightness_analysis.classify(features,labels,ID,ID2,threshold)


# In[5]:


print(classification_report(results[0],labels))
# Threshold = 0.012 seems to be the best


# In[6]:


print("Kappa =",cohen_kappa_score(results[0],labels))


# In[26]:


a=0
b=0
c=0
d=0

for i in range(len(labels)):
    if labels[i] == 1:
#         print(labels[i])
#         print(results[0][i])
#         print('----------')
        if labels[i] != results[0][i]:
            a+=1
#             print('W =',a,i)
        elif labels[i] == results[0][i]:
            b+=1
#             print('G =',b,i)

    if labels[i] == 0:
            if labels[i] != results[0][i]:
                c+=1
            elif labels[i] == results[0][i]:
                d+=1


# In[27]:


print('Wt=',a) # Wrong true sats   ->    True sats which were wrongly classified
print('Gt =',b) # Good true sats   ->    True sats which were correctly classified
print('Wf =',c) # Wrong false sats   ->    False sats which were wrongly classified
print('Gf =',d) # Good false sats   ->    False sats which were correctly classified


# In[9]:


img = features[3506]
plt.imshow(img.reshape((7,7)), cmap = 'gray', vmin=0, vmax=1)


# #### One variation of the algorithm is to look for 4 pixels anywhere in area1 which have an average brightness which deviates the most from the average of the last ring of the kernel or area2.
# #### If any pixel located in area2 exceeds the avg of area2 without counting that pixel by a certain margin, ignore it and set it equal to the avg of area2
# #### If the threshold is adjusted to catch all the true sats then all the false positives can be filtered (in the dataset images) by integrating the expected motion of the GEOsats from frame to frame

# In[10]:


example_image=read_image(path3)
plt.figure()
plt.imshow(example_image, cmap = 'gray', vmin=0, vmax=1)
plt.axis('off')
plt.xlim(500,620)
plt.ylim(100,200)
plt.savefig('img2.png',bbox_inches='tight', dpi=50)


# In[11]:


path4 = 'C:/Users/sincl/OneDrive/Desktop/Cranfield University/spotGEO project/Dataset/Latest work/img2.png'
img2 = read_image(path4)


# In[12]:


plt.imshow(img2)


# In[13]:


print(img2.shape)


# In[14]:


print(img2[0][0])


# In[15]:


img2 = img2.tolist()


# In[16]:


for i in range(len(img2)):
    for j in range(len(img2[0])):
        img2[i][j] = sum(img2[i][j])/len(img2[i][j])
        if img2[i][j] == 1:
            img2[i][j] = 0.373


# In[17]:


img2 = np.asarray(img2)


# In[18]:


plt.imshow(img2, cmap = 'gray', vmin=0, vmax=1)


# In[19]:


print(img2.shape)
simulated_labels = []

for i in range(25696):
    simulated_labels.append(np.random.randint(0,2))
    
simulated_labels = np.asarray(simulated_labels)


# In[20]:


def get_kernel(img):
    kernel = []
    for u in range(len(img)):
        for t in range(len(img[0])):
            if u > 7 and u < 154 and t > 7 and t < 184:
                kernel.append(list(itertools.chain(*[img[u-3][t-3:t+4],img[u-2][t-3:t+4],img[u-1][t-3:t+4],img[u][t-3:t+4],img[u+1][t-3:t+4],img[u+2][t-3:t+4],img[u+3][t-3:t+4]])))

            
#     kernel = list(itertools.chain(*kernel))
#     for j in range(len(kernel)):
#         kernel[j] = int(kernel[j])
        
    return np.asarray(kernel)


# In[21]:


img_features = get_kernel(img2)


# In[22]:


print(len(img_features))
print(len(simulated_labels))


# In[23]:


results2 = []
threshold = 0.015
ID = 0
ID2 = 3

results2.append(Brightness_analysis.classify(get_kernel(img2),simulated_labels,ID,ID2,threshold))


# In[24]:


a = 0
b = 0
for i in range(len(results2[0][0])):
    if results2[0][0][i] == 1:
        a+=1
#         print(results2[0][1][i])
    elif results2[0][0][i] == 0:
        b+=1
print(a,b)


# In[ ]:




