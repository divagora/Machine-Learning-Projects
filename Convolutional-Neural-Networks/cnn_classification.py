#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 13:40:22 2020

@author: ls616
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import imutils

import os
from os import listdir

from distutils.dir_util import copy_tree
from shutil import rmtree

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore")




## set wd ##
os.chdir('/Users/ls616/Google Drive/MPE CDT/ML Projects/Projects/CNN/')



### DATA AUGMENTATION ###

## exploratory analysis of data ##

## fn. to count no. of samples (original OR augmented OR both) ##
def count_samples(dir_list,aug_dir_list,sample_type="original"):
    
    if sample_type=="original" or sample_type=="both":
        n_samples = []
        for i in dir_list:
            n_samples.append(len([name for name in listdir(i) if not name.startswith(".")]))
        n_total_samples = np.sum(n_samples)
        
        print(f"\nThere are {n_total_samples} original samples in total")
        print(f"There are {n_samples[0]} ({round(n_samples[0]/n_total_samples*100,2)} %) original samples with tumours")
        print(f"There are {n_samples[1]} ({round(n_samples[1]/n_total_samples*100,2)} %) original samples without tumours\n") 
        
        if sample_type=="original":
            return n_samples, n_total_samples
        
    
    if sample_type=="augmented" or sample_type=="both":
        n_aug_samples = []
        for i in aug_dir_list:
            n_aug_samples.append(len([name for name in listdir(i) if not name.startswith(".")]))
        n_total_aug_samples = np.sum(n_aug_samples)
        
        print(f"\nThere are {n_total_aug_samples} augmented samples in total")
        print(f"There are {n_aug_samples[0]} ({round(n_aug_samples[0]/n_total_aug_samples*100,2)} %) augmented samples with tumours")
        print(f"There are {n_aug_samples[1]} ({round(n_aug_samples[1]/n_total_aug_samples*100,2)} %) augmented samples without tumours\n")  
    
        if sample_type=="augmented":
            return n_aug_samples, n_aug_total_samples
        
        
    if sample_type=="both":
        n_all_samples = [sum(x) for x in zip(n_samples,n_aug_samples)]
        n_all_total_samples = n_total_samples + n_total_aug_samples
        
        print(f"\nThere are {n_all_total_samples} samples in total")
        print(f"There are {n_all_samples[0]} ({round(n_all_samples[0]/n_all_total_samples*100,2)} %) samples with tumours")
        print(f"There are {n_all_samples[1]} ({round(n_all_samples[1]/n_all_total_samples*100,2)} %) samples without tumours\n")  
        
        return n_samples, n_total_samples, n_aug_samples, n_aug_total_samples, n_all_samples, n_all_total_samples


## count samples ##
dir_list = ['og_data/yes','og_data/no']
n_samples, n_total_samples = count_samples(dir_list=dir_list,aug_dir_list=None,sample_type="original")

## fn. to augment data ##
def augment_data(dir_list, n_new_samples, aug_dir_list):
    
    
    data_gen = ImageDataGenerator(rotation_range=10, 
                                  width_shift_range=0.1, 
                                  height_shift_range=0.1, 
                                  shear_range=0.1, 
                                  brightness_range=(0.3, 1.0),
                                  horizontal_flip=True, 
                                  vertical_flip=True, 
                                  fill_mode='nearest'
                                 )
    
            
    for i in range(len(dir_list)):
        
        ## delete any existing files in target
        rmtree(aug_dir_list[i])
        os.mkdir(aug_dir_list[i])
        
        for j in listdir(dir_list[i]):
            
            # load image
            img = cv2.imread(dir_list[i]+'/'+j)
            
            # reshape image
            img = img.reshape((1,)+img.shape)
            
            # save directory 
            save_to_dir = aug_dir_list[i]
            
            # save prefixed
            save_prefix = 'aug_' + j[:-4]
        
            # generate 'n_generated_samples' new samples
            count = 1
            for batch in data_gen.flow(x=img, batch_size=1, 
                                   save_to_dir=save_to_dir, 
                                   save_prefix=save_prefix, 
                                   save_format='jpg'):
                count += 1
                if count > n_new_samples[i]:
                    break


## augment data ##
## to balance data, we generate:
## -> 1 new samples for each image with a tumour
## -> 2 new samples for each image without a tumour 
n_new_samples = [12,19]
aug_dir_list = ['aug_data/yes','aug_data/no']
augment_data(dir_list,n_new_samples,aug_dir_list)

## count no. of augmented samples ##
n_aug_samples, n_aug_total_samples = count_samples(dir_list,aug_dir_list,sample_type="augmented")

## count no. of all (original and augmented) samples ##
n_samples, n_total_samples, n_aug_samples, n_aug_total_samples, n_all_samples, n_all_total_samples = count_samples(dir_list,aug_dir_list,"both")


## merge all data in single folder
all_dir_list = ['all_data/yes','all_data/no']
for i in range(2):
    
    ## delete any existing files in target
    rmtree(all_dir_list[i])
    os.mkdir(all_dir_list[i])
    
    ## copy original data
    copy_tree(dir_list[i], all_dir_list[i])
    
    ## copy augmented data
    copy_tree(aug_dir_list[i],all_dir_list[i])



### MODEL FITTING ###

## fn. to crop images ##
## see https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/ ##
def crop_img(img, plot=False):
    
    # Convert image to grayscale, add blur 
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.GaussianBlur(bw, (5, 5), 0)

    # Threshold image, use erosions + dilations to remove regions of noise
    thrsh = cv2.threshold(bw.copy(), 45, 255, cv2.THRESH_BINARY)[1]
    thrsh = cv2.erode(thrsh, None, iterations=2)
    thrsh = cv2.dilate(thrsh, None, iterations=2)

    # Find contours in thresholded image, select the largest contour
    cnts = cv2.findContours(thrsh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find extreme points
    ext_left = tuple(c[c[:, :, 0].argmin()][0])
    ext_right = tuple(c[c[:, :, 0].argmax()][0])
    ext_top = tuple(c[c[:, :, 1].argmin()][0])
    ext_bot = tuple(c[c[:, :, 1].argmax()][0])
    
    # Crop the new image out of original image using the extreme points
    new_img = img[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]            

    if plot:
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.tick_params(axis='both', which='both', 
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        
        plt.title('Original Image')
        
        plt.subplot(1, 2, 2)
        plt.imshow(new_img)
        plt.tick_params(axis='both', which='both', 
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.title('Cropped Image')
        plt.show()
    
    return new_img


## test on single image ##
img_test = cv2.imread('all_data/yes/Y1.jpg')
crop_img(img_test,plot=True)


## fn. to import images ##
def import_data(img_size,dir_list):
    
    ## initialise inputs and outputs
    X, y = [], []
    
    for i in dir_list:
        for j in listdir(i):
            
            ## load image
            img = cv2.imread(i+"/"+j)
            
            ## crop image ##
            img = crop_img(img)
            
            ## resize image
            img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
            
            ## normalise values
            img = img/255
            
            ## append to X
            X.append(img)
            
            ## append classification value to y
            if i[-3:]=="yes":
                y.append(1)
            if i[-2:]=="no":
                y.append(0)
         
                
    ## convert to np array
    X = np.array(X)
    y = np.array(y)
    
    ## shuffle data
    X, y = shuffle(X,y)
    
    ## summarise data
    print(f'The number of samples is {len(X)}')
    print(f'The size of X is {X.shape}')
    print(f'The size of y is {y.shape}')
    
    return X, y

## import data ##
img_size = (224,224); dir_list = ['all_data/yes','all_data/no']
X,y = import_data(img_size,dir_list)


## fn. to plot several images ##
def plot_images(X, y, n=8):
    
    ## first n imgs with y=0,1 ##
    for y_label in [0,1]:
        imgs = X[np.argwhere(y == y_label)]
        n_imgs = imgs[:n]
        
        ## arrange images
        col_n = 4
        row_n = int(n/col_n)
        plt.figure(figsize=(col_n*2, row_n*2+.1))
        
        
        counter = 1 # current plot        
        for image in n_imgs:
            plt.subplot(row_n, col_n, counter)
            plt.imshow(image[0])
            
            # remove ticks
            plt.tick_params(axis='both', which='both', 
                            top=False, bottom=False, left=False, right=False,
                           labelbottom=False, labeltop=False, labelleft=False, 
                           labelright=False)
            
            counter += 1
        
        label_to_str = lambda x: "" if x == 1 else "No"
        plt.suptitle(f"{label_to_str(y_label)} Brain Tumor")
        plt.show()        
    
## plot images ##
plot_images(X,y,8)



## training and test data ##
def train_test_val_split(X,y,val_size,test_size):
    
    ## split all data into training data & validation/test data
    X_train,X_test_val,y_train,y_test_val = train_test_split(X,y,test_size = val_size+test_size)
    
    ## split validation/test data validation data & test data
    X_val,X_test,y_val,y_test = train_test_split(X_test_val,y_test_val,test_size = test_size/(test_size+val_size))
    
    ## print summary
    print(f"\nThe total no. of samples is {X.shape[0]}")
    print (f"The no. of training samples is {X_train.shape[0]}")
    print (f"The no. of validation samples is {X_val.shape[0]}")
    print (f"The no. of test samples is {X_test.shape[0]}\n")
    
    return X_train,X_val,X_test,y_train,y_val,y_test
    
X_train,X_val,X_test,y_train,y_val,y_test = train_test_val_split(X,y,val_size=0.15,test_size=0.15)
    

        
    
## fn. to build model ##
def build_model(input_shape):
    
    ## initialise model
    model = Sequential()
    
    ## zero padding
    model.add(ZeroPadding2D((2,2), input_shape = input_shape))
    
    ## conv2d layer
    model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0'))
    
    ## batch normalisation layer
    model.add(BatchNormalization(axis = 3, name = 'bn0'))
    
    ## activation
    model.add(Activation('relu')) # shape=(?, 238, 238, 32)
    
    # max pooling
    model.add(MaxPooling2D((4, 4), name='max_pool0')) # shape=(?, 59, 59, 32) 
    
    # max pooling
    model.add(MaxPooling2D((4, 4), name='max_pool1')) # shape=(?, 14, 14, 32)
    
    # flatten 
    model.add(Flatten()) # shape=(?, 6272)
    
    # dense
    model.add(Dense(1, activation='sigmoid', name='fc')) # shape=(?, 1)
    
    print(model.summary())
    
    return model
        

## build model 
input_shape = X_train.shape[1:]
model = build_model(input_shape)

## compile model
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

## train model
history = model.fit(x=X_train, y=y_train, batch_size=32, epochs=50, 
                    validation_data=(X_val, y_val),verbose=1)        
        