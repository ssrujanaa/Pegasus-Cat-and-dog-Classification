#!/usr/bin/env python3
# coding: utf-8
from os import listdir
from numpy import asarray
from numpy import save
from glob import glob
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import sys
import os


def replace_filename(input, pattern, replaceWith): 
    return input.replace(pattern, replaceWith) 

def main():
    input_files = glob('*.jpg')
    photos, labels = list(), list()
    cat_files =[]
    dog_files =[]

    for i in range(len(input_files)):
        if 'Cat' in input_files[i]:
            cat_files.append(input_files[i])
            output = 1.0
            photo = load_img(input_files[i], target_size=(200, 200))
            photo.save(replace_filename(input_files[i],'Cat','resized_Cat'), 'JPEG', quality=90)
            photos.append(photo)
            labels.append(output)
        else:
            dog_files.append(input_files[i]) 
            output = 0.0
            photo = load_img(input_files[i], target_size=(200, 200))
            photo.save(replace_filename(input_files[i],'Dog','resized_Dog'), 'JPEG', quality=90)
            photos.append(photo)
            labels.append(output)  

    with open('resized_images.txt', 'w') as f:
        for item in photos:
            f.write("%s\n" % item)

    with open('labels.txt', 'w') as f:
        for item in labels:
            f.write("%s\n" % item)
    return 0

if __name__ == '__main__':
    main()