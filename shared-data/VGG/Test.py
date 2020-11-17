#!/usr/bin/env python3
# coding: utf-8

#Predicting the model performance on test dataset
import pickle
import signal
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from keras import layers
from keras.layers import Input,Dense,BatchNormalization,Flatten,Dropout,GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.utils import layer_utils
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
import keras.backend as K
import traceback
from keras.applications.vgg16 import VGG16
from keras.models import Model,load_model
import pandas as pd
import h5py
import sys
import joblib
import argparse
import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

def get_test_data():
    with open('testing.pkl', 'rb') as f:
         test = pickle.load(f)
            
    test_photos, test_labels = list(), list()
    for file in test:
        if 'Cat' in file:
            output = 1.0
        else:
            output = 0.0
        photo = load_img(file)
        photo = img_to_array(photo)
        test_photos.append(photo)
        test_labels.append(output)
    test_photos = asarray(test_photos)
    test_labels = asarray(test_labels)
    
    return test_photos,test_labels

model = load_model('model.h5')
test_photos, test_labels = get_test_data()
# test_acc = model.evaluate(test_photos, test_labels, batch_size=2)
#Test Accuracy is test_acc[1]

# predict probabilities for test set
yhat_probs = model.predict(test_photos, verbose=0)
# predict crisp classes for test set
yhat_classes = yhat_probs.argmax(axis=-1)
# reduce to 1d array
yhat_probs = yhat_probs[:][0]
yhat_classes = yhat_classes[:]

accuracy = accuracy_score(test_labels, yhat_classes)
print('Accuracy: %f' % accuracy)

precision = precision_score(test_labels, yhat_classes)
print('Precision: %f' % precision)

recall = recall_score(test_labels, yhat_classes)
print('Recall: %f' % recall)

f1 = f1_score(test_labels, yhat_classes)
print('F1 score: %f' % f1)

matrix = confusion_matrix(test_labels, yhat_classes)
print(matrix)

output_file = 'Result_Metrics.txt'
with open(output_file, 'w') as f:
    f.write("Test Accuracy:" + str(accuracy) + "\n" + "Precision:" + str(precision) + "\n" + "Recall:" + str(recall) + "\n"
           + "F1 score:" + str(f1) + "\n" + "Confusion Matrix:" + str(list(matrix)) )
