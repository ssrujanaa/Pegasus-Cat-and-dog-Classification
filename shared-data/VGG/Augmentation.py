#!/usr/bin/env python3
# coding: utf-8
from keras.preprocessing.image import ImageDataGenerator,  array_to_img, img_to_array, load_img 
from glob import glob
import os

def augment():
    resized_images = glob('resized_*.jpg')
    labels_txt = 'labels.txt'
    cat_images=[]
    dog_images=[]
    labels=[] 
    q = []
    for i in range(len(resized_images)):
        if 'resized_Cat' in resized_images[i]:
            cat_images.append(resized_images[i])
        else:
            dog_images.append(resized_images[i])

    datagen = ImageDataGenerator( rescale=1./255,
            rotation_range = 40,  
            horizontal_flip = True, 
            brightness_range = (0.5, 1.5)) 

    for image in cat_images:
        num = 1.0
        img = load_img(image)  
        x = img_to_array(img) 
        x = x.reshape((1, ) + x.shape)  
        # Generating and saving 5 augmented samples  
        i = 0
        for batch in datagen.flow(x, batch_size = 1,
                                  save_to_dir = os.getcwd(),  
                                  save_prefix ='Aug_' + os.path.splitext(image)[0] + '_' + str(i), save_format ='jpg'): 
            i += 1
            labels.append(num)
            if i > 2: 
                break

    for image in dog_images:
        num = 1.0
        img = load_img(image)  
        x = img_to_array(img) 
        x = x.reshape((1, ) + x.shape)  
        # Generating and saving 5 augmented samples  
        i = 0
        for batch in datagen.flow(x, batch_size = 1,
                                  save_to_dir = os.getcwd(),  
                                  save_prefix= 'Aug_' + os.path.splitext(image)[0] + '_' + str(i), save_format ='jpg'): 
            i += 1
            labels.append(num)
            if i > 2: 
                break
    return cat_images, dog_images,q,labels
                
                
def rename_augmented_images():
    cat_images,dog_images,q,labels = augment()
    aug_cat = glob('Aug_resized_Cat*.jpg')
    aug_dog = glob('Aug_resized_Dog*.jpg')
    aug_images = aug_cat + aug_dog
    aug_cat.sort()
    for n in range(len(cat_images)):
        k = 0
        for m in range(len(aug_cat)):
            if os.path.splitext(cat_images[n])[0] in aug_cat[m]:
                photo = load_img(aug_cat[m])
                photo.save('Aug_'+ os.path.splitext(cat_images[n])[0] + '_' + str(k) + '.jpg', 'JPEG', quality=90)
                q.append('Aug_'+ os.path.splitext(cat_images[n])[0] + '_' + str(k) + '.jpg')
                k+=1

    aug_dog.sort()
    for n in range(len(dog_images)):
        k = 0
        for m in range(len(aug_dog)):
            if os.path.splitext(dog_images[n])[0] in aug_dog[m]:
                photo = load_img(aug_dog[m])
                photo.save('Aug_'+ os.path.splitext(dog_images[n])[0] + '_' + str(k) + '.jpg', 'JPEG', quality=90)
                q.append('Aug_'+ os.path.splitext(dog_images[n])[0] + '_' + str(k) + '.jpg')
                k+=1 
    return aug_images,labels

def main():
    aug_images,labels = rename_augmented_images()
    aug_images_txt = 'augmentation.txt'
    aug_labels_txt = 'aug_labels.txt'
    with open(aug_images_txt, 'w') as f:
        for item in aug_images:
            f.write("%s\n" % item)

    with open(aug_labels_txt, 'w') as f:
        for item in labels:
            f.write("%s\n" % item)
    return 0

if __name__ == '__main__':
    main()