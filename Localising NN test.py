#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function;
import cv2;
import numpy as np;
import matplotlib.pyplot as plt;
import random;
import tqdm;
import keras
import tensorflow;
from tensorflow.keras import layers;
from tensorflow.keras import Model;


# In[6]:


IMG_SIZE = 200;
rad = random.randint(1,10);
cx = random.randint(rad,IMG_SIZE-rad);
cy = random.randint(rad, IMG_SIZE-rad);
blank_image = np.ones(shape = [IMG_SIZE, IMG_SIZE], dtype = np.uint8);
new_img = cv2.circle(blank_image, (cx,cy), rad, 0, -1);

plt.imshow(new_img)
print(np.shape(new_img))


# In[3]:


def create_training_data():
    l = 10000;
    X_train = np.zeros(shape=[l, IMG_SIZE, IMG_SIZE,1]);
    Y_train = np.zeros(shape = [l,3]);
    
    for i in range(l):
        rad = random.randint(1,10);
        cx = random.randint(rad,IMG_SIZE-rad);
        cy = random.randint(rad, IMG_SIZE-rad);
        Y_train[i,0] = cx/IMG_SIZE;
        Y_train[i,1] = cy/IMG_SIZE;
        Y_train[i,2] = rad/IMG_SIZE;
        blank_image = np.ones(shape=[IMG_SIZE, IMG_SIZE], dtype = np.uint8);
        X_train[i,:,:,0] = cv2.circle(blank_image, (cx,cy), rad, 0, -1);
        
    return {'X_Train' : X_train, 'Y_train' : Y_train};


# In[4]:


training_data = create_training_data();


# In[5]:


plt.imshow(training_data['X_Train'][1].reshape(200,200))
200*training_data['Y_train'][1]


# In[6]:


img_input = layers.Input(shape = (IMG_SIZE, IMG_SIZE,1))

x = layers.Conv2D(5, 3,
                 activation='relu',
                 strides = 1,
                 padding = 'same')(img_input)
x = layers.MaxPool2D(pool_size = 2)(x)

x = layers.Conv2D(10, 3,
                 activation='relu',
                 strides = 2)(x)
x = layers.MaxPool2D(pool_size=2)(x)

x = layers.Conv2D(20, 3,
                 activation='relu',
                 strides = 2)(x)
x = layers.MaxPool2D(pool_size=2)(x)

x = layers.Conv2D(3, 5,
                 activation='relu',
                 strides = 2)(x)

output = layers.Flatten()(x)

model = Model(img_input, output)

model.summary()


# In[7]:


model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse']);

hist = model.fit(training_data["X_Train"], training_data["Y_train"],
          epochs=1, verbose=1)


# In[8]:


(IMG_SIZE)*model.predict(training_data['X_Train'][2].reshape(1, IMG_SIZE, IMG_SIZE,1))


# In[9]:


IMG_SIZE*training_data['Y_train'][2]


# In[10]:


for i in range(10):
    print('prediction =', IMG_SIZE*model.predict(training_data['X_Train'][i].reshape(1, IMG_SIZE, IMG_SIZE,1)))
    print('actual =', IMG_SIZE*training_data['Y_train'][i])
    print()


# In[ ]:




