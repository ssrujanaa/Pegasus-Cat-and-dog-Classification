#!/usr/bin/env python3
# coding: utf-8
from sklearn.model_selection import train_test_split
from glob import glob
import pickle

def split_data():
    input_images = glob('Aug_resized_*.jpg')
    cat = []
    dog = []
    for i in range(len(input_images)):
        if 'resized_Cat' in input_images[i]:
            cat.append(input_images[i])
        else:
            dog.append(input_images[i]) 

    train_cat, test_cat = train_test_split(cat, test_size=0.2, random_state=42, shuffle=True)
    train_dog, test_dog = train_test_split(dog, test_size=0.2, random_state=42, shuffle=True)

    training_cat, val_cat = train_test_split(train_cat, test_size=0.1, random_state=42, shuffle=True)
    training_dog, val_dog = train_test_split(train_dog, test_size=0.1, random_state=42, shuffle=True)
    
    training_data = training_cat + training_dog
    testing_data = test_cat + test_dog
    val_data = val_cat + val_dog
    return training_data, testing_data,val_data

def main():
    training_data, testing_data,val_data = split_data()
    with open('training.pkl', 'wb') as f:
        pickle.dump(training_data, f)

    with open('testing.pkl', 'wb') as f:
        pickle.dump(testing_data, f)

    with open('validation.pkl', 'wb') as f:
        pickle.dump(val_data,f)
    return 0

if __name__ == '__main__':
    main()