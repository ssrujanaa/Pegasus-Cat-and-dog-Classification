#!/usr/bin/env python3
# coding: utf-8
import sys
import shutil
import pickle
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
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model,load_model
from optkeras.optkeras import OptKeras
import optkeras
import pickle
from keras.optimizers import RMSprop
import optuna
import os
import tensorflow as tf
import argparse
import joblib
import pandas as pd
optkeras.optkeras.get_trial_default = lambda: optuna.trial.FrozenTrial(
        None, None, None, None, None, None, None, None, None, None, None)


import keras
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

    
def hpo_monitor(study, trial):
    joblib.dump(study,"hpo_checkpoint.pkl")
    
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

def objective(trial):
    train_photos,train_labels,test_photos,test_labels,val_photos,val_labels = get_data()
    nb_classes = 2
    epochs=1
    batch_size =5
    optimizer_options = ["RMSprop", "Adam", "SGD"]

    vgg16_model = VGG16(weights = 'imagenet', include_top = False)
    x = vgg16_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation=trial.suggest_categorical('activation', ['relu', 'linear']))(x)
    predictions = Dense(nb_classes, activation = 'softmax')(x)
    model = Model(input = vgg16_model.input, output = predictions)
    for layer in vgg16_model.layers:
        layer.trainable = False

    model.compile(optimizer = trial.suggest_categorical("optimizer", ["rmsprop", "Adam", "SGD"]),loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    model.fit(x=train_photos, y=train_labels, batch_size=2 , epochs=epochs, validation_data=(val_photos,val_labels))
    return 0


def main():
    hpo_checkpoint_file = 'hpo_checkpoint.pkl'
    N_TRIALS = 6
    tune_from_beginning = False
    try:
        ok = joblib.load(hpo_checkpoint_file)
        todo_trials = N_TRIALS - len(ok.trials_dataframe())
        if todo_trials > 0 :
            ok.optimize(objective, n_trials=todo_trials, timeout=600, callbacks=[hpo_monitor])
        else:
            pass
    except KeyError:
        tune_from_beginning = True

    if tune_from_beginning: 
        study_name = "CatsAndDogs" + '_Simple'
        ok = OptKeras(study_name=study_name,monitor='val_acc',direction='maximize')
        ok.optimize(objective, n_trials=N_TRIALS, timeout=600, callbacks=[hpo_monitor])
    output_file = 'hpo_results.pkl'
    shutil.copyfile(hpo_checkpoint_file, output_file)
    
    return 0
        
if __name__ == '__main__':
    main()