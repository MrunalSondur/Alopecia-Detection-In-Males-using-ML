#!/usr/bin/env python
# coding: utf-8

# # CV_CP_Dataset only 

# In[32]:


import cv2
import numpy as np
import pandas as pd
import pickle
import os
import csv
import matplotlib.pyplot as plt
## pip install split-folders
import splitfolders 


# In[33]:


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


# In[34]:


input0=r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_1\T1_HairBald"
input1=r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_2\T2_Hairbald"
input2=r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_3\T3_HairBald"
input3=r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_4\T4_HairBald"


# In[35]:


### For Dataset TYPE 1: 
# For Class of type 1 with index i = 0

i=0
for filename in os.listdir(input0):
   
    path=os.path.join(input0,filename)
    in0=cv2.imread(path)
    path
    print("Input Image : ", input0, i)

    plt.imshow(in0, cmap = 'gray')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Input Gray Image')
    plt.show()
    
    #resize image
    resize=(120,120)
    img0=cv2.resize(in0,resize)
    gray0=cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

        
    plt.imshow(gray0, cmap = 'gray')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Resized Image:')
    plt.show()
    
    #gray image
    #path= r"./fruitDataset/preproc_dataset/PreApple/"
    path= r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_1\T1_Pre_hairbald"
    cv2.imwrite(os.path.join(path, 't1hb'+ str(i) + '.jpg'), gray0)
    i=i+1

    cv2.waitKey(0)


# In[36]:


### For dataset: TYPE 2

i=0
for filename in os.listdir(input1):
   
    path=os.path.join(input1,filename)
    in1=cv2.imread(path)
    path
    print("Input Image : ", input1, i)

    plt.imshow(in1, cmap = 'gray')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Input Gray Image')
    plt.show()
    
    #resize image
    resize=(120,120)
    img1=cv2.resize(in0,resize)
    gray1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        
    plt.imshow(gray1, cmap = 'gray')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Resized Image:')
    plt.show()
    
    #gray image
    #path= r"./fruitDataset/preproc_dataset/PreApple/"
    path= r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_2\T2_Pre_hairbald"
    cv2.imwrite(os.path.join(path, 't2hb'+ str(i) + '.jpg'), gray1)
    i=i+1

    cv2.waitKey(0)


# In[37]:


### For dataset: TYPE 3

i=0
for filename in os.listdir(input2):
   
    path=os.path.join(input2,filename)
    in2=cv2.imread(path)
    path
    print("Input Image : ", input2, i)

    plt.imshow(in2, cmap = 'gray')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Input Gray Image')
    plt.show()
    
    #resize image
    resize=(120,120)
    img2=cv2.resize(in2,resize)
    gray2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        
    plt.imshow(gray2, cmap = 'gray')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Resized Image:')
    plt.show()
    
    #gray image
    #path= r"./fruitDataset/preproc_dataset/PreApple/"
    path= r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_3\T3_Pre_hairbald"
    cv2.imwrite(os.path.join(path, 't3hb'+ str(i) + '.jpg'), gray2)
    i=i+1

    cv2.waitKey(0)


# In[38]:


### For dataset: TYPE 4

i=0
for filename in os.listdir(input3):
   
    path=os.path.join(input3,filename)
    in3=cv2.imread(path)
    path
    print("Input Image : ", input3, i)

    plt.imshow(in3, cmap = 'gray')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Input Gray Image')
    plt.show()
    
    #resize image
    resize=(120,120)
    img3=cv2.resize(in3,resize)
    gray3=cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

        
    plt.imshow(gray3, cmap = 'gray')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Resized Image:')
    plt.show()
    
    #gray image
    #path= r"./fruitDataset/preproc_dataset/PreApple/"
    path= r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_4\T4_Pre_hairbald"
    cv2.imwrite(os.path.join(path, 't4hb'+ str(i) + '.jpg'), gray3)
    i=i+1

    cv2.waitKey(0)


# ### ------------------------------------------------------------------------------------------------------------------------------------------------
# ### The below for Data Augmentation 

# In[39]:


datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15, # .1, .2, .3
    width_shift_range=0.2,
    height_shift_range=0.2,
    #rescale=1./255,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")


# In[40]:


### TYPE 1 ###

input0 =  r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_1\T1_Pre_hairbald"

i=0
for filename in os.listdir(input0):
   
    path=os.path.join(input0,filename)
    pic = load_img(path)
    pic.getpixel
    pic_array = img_to_array(pic)
    pic_array = pic_array.reshape((1,) + pic_array.shape)

    count = 0
    output = r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_1\T1_AugGen"
    for batch in datagen.flow( pic_array, batch_size = 1, save_to_dir=output,  save_prefix='HBT1', save_format='jpg'):
        count += 1
        if count > 10:
            break


# In[41]:


### TYPE 2 ###

input0 =  r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_2\T2_Pre_hairbald"

i=0
for filename in os.listdir(input0):
   
    path=os.path.join(input0,filename)
    pic = load_img(path)
    pic.getpixel
    pic_array = img_to_array(pic)
    pic_array = pic_array.reshape((1,) + pic_array.shape)

    count = 0
    output = r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_2\T2_AugGen"
    for batch in datagen.flow( pic_array, batch_size = 1, save_to_dir=output,  save_prefix='HBT2', save_format='jpg'):
        count += 1
        if count > 10:
            break


# In[42]:


### TYPE 3 ###

input0 =  r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_3\T3_Pre_hairbald"

i=0
for filename in os.listdir(input0):
   
    path=os.path.join(input0,filename)
    pic = load_img(path)
    pic.getpixel
    pic_array = img_to_array(pic)
    pic_array = pic_array.reshape((1,) + pic_array.shape)

    count = 0
    output = r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_3\T3_AugGen"
    for batch in datagen.flow( pic_array, batch_size = 1, save_to_dir=output,  save_prefix='HBT3', save_format='jpg'):
        count += 1
        if count > 10:
            break


# In[43]:


### TYPE 4 ###

input0 =  r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_4\T4_Pre_hairbald"

i=0
for filename in os.listdir(input0):
   
    path=os.path.join(input0,filename)
    pic = load_img(path)
    pic.getpixel
    pic_array = img_to_array(pic)
    pic_array = pic_array.reshape((1,) + pic_array.shape)

    count = 0
    output = r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_4\T4_AugGen"
    for batch in datagen.flow( pic_array, batch_size = 1, save_to_dir=output,  save_prefix='HBT4', save_format='jpg'):
        count += 1
        if count > 10:
            break


# In[ ]:




