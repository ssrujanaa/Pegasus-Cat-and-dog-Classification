#!/usr/bin/env python3
# coding: utf-8
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

def parse_args(args):
    parser = argparse.ArgumentParser(description='Cat and Dog image classification using Keras')
    parser.add_argument('-epochs',  metavar='num_epochs', type=int, default = 5, help = "Number of training epochs")
    parser.add_argument('--batch_size',  metavar='batch_size', type=int, default = 16, help = "Batch Size")
    parser.add_argument('-f')
    return parser.parse_args()

#get training, testing and validation data from the saved pickle files.
def get_data():
    with open('training.pkl', 'rb') as f:
         train = pickle.load(f)

    with open('testing.pkl', 'rb') as f:
         test = pickle.load(f)

    with open('validation.pkl','rb') as f:
        val = pickle.load(f)

    train_photos, train_labels = list(), list()
    tp = list()
    for file in train:
        if 'Cat' in file:
            output = 1.0
        else:
            output = 0.0
        photo = load_img(file)
        photo = img_to_array(photo)
        train_photos.append(photo)
        train_labels.append(output)
    train_photos = asarray(train_photos)
    train_labels = asarray(train_labels)

    test_photos, test_labels = list(), list()
    for file in test:
        if 'Cat' in file:
            output = 1.0
        else:
            output = 0.0
        photo = load_img(file)
        photo = img_to_array(photo)
        tp.append(photo)
        test_photos.append(photo)
        test_labels.append(output)
    test_photos = asarray(test_photos)
    test_labels = asarray(test_labels)

    val_photos, val_labels = list(), list()
    for file in val:
        if 'Cat' in file:
            output = 1.0
        else:
            output = 0.0
        photo = load_img(file)
        photo = img_to_array(photo)
        val_photos.append(photo)
        val_labels.append(output)
    val_photos = asarray(val_photos)
    val_labels = asarray(val_labels)
    return train_photos,train_labels,test_photos,test_labels,val_photos,val_labels

#Definition of the VGG16 model and changing the output layer according to our requirements.
#i.e., 2 output classes
def get_model():
    nb_classes = 2
    Study = joblib.load('hpo_results.pkl')
    _dict = Study.best_trial.params
    activation_optuna = _dict['activation']
    optimizer_optuna = _dict['optimizer']
    
    vgg16_model = VGG16(weights = 'imagenet', include_top = False)
    x = vgg16_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(nb_classes, activation = activation_optuna)(x)
    model = Model(input = vgg16_model.input, output = predictions)

    for layer in vgg16_model.layers:
        layer.trainable = False
    model.compile(optimizer = optimizer_optuna,loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return model,optimizer_optuna


def main():
    train_photos,train_labels,test_photos,test_labels,val_photos,val_labels = get_data()
    model,optimizer_optuna = get_model()
    
    #checkpoint file that saves the weights after each epoch - weights are overwritten to the same file
    checkpoint_file = 'checkpoint_file2.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss', verbose=1, mode='auto',save_weights_only = True, period=1)
    
    train_from_beginning = False
    try:
        #Since our hdf5 file contains additional data = epochs, skip_mismatch is used to avoid that column
        model.load_weights("checkpoint_file2.hdf5",skip_mismatch=True)
        with h5py.File('checkpoint_file2.hdf5', "r+") as file:
            data = file.get('epochs')[...].tolist()
            
        #loading the number of epochs already performed to resume training from that epoch
        initial_epoch = data
        model.compile(optimizer = optimizer_optuna,loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        for i in range(initial_epoch,EPOCHS):
            model.fit(x=train_photos, y=train_labels,batch_size=BATCH_SIZE , epochs=1, verbose=1,
                      validation_data=(val_photos,val_labels), callbacks = [checkpoint])
            checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss', verbose=1, mode='auto',
                                         save_weights_only = True, period=1)
            
            #saving the number of finished epochs to the same hdf5 file
            with h5py.File('checkpoint_file2.hdf5', "a") as file:
                file['epochs'] = i
    except OSError:
        train_from_beginning = True

    if train_from_beginning:
        model.compile(optimizer = 'rmsprop',loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        for i in range(EPOCHS):
            model.fit(x=train_photos, y=train_labels,batch_size=BATCH_SIZE , epochs=1, 
                           verbose=1,validation_data=(val_photos,val_labels), callbacks = [checkpoint])
            checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss', verbose=1, mode='auto',save_weights_only = True, period=1)
            #saving the number of finished epochs to the same hdf5 file
            with h5py.File('checkpoint_file2.hdf5', "a") as file:
                file['epochs']=i

    model.save('model.h5')
    return 0
    
if __name__ == '__main__':
    global EPOCHS
    global BATCH_SIZE
    args = parse_args(sys.argv[1:])
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    main()