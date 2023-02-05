from os.path import getsize
from PIL import Image
import numpy as np
import os
from random import shuffle
import imageio
import matplotlib.pyplot as plt
from numpy.lib.function_base import delete
import seaborn as sns
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


imgDIR = ".\\images"




print("Mapping image ID field in training csv file to an image category") 
naming_dict_train = {}
fTr = open(".\\train.csv", "r")
fileContentsTrain = fTr.read()
fileContentsTrain = fileContentsTrain.split('\n')
for i in range(len(fileContentsTrain)-1):
  fileContentsTrain[i] = fileContentsTrain[i].split('\t')
  naming_dict_train[fileContentsTrain[i][0]] = fileContentsTrain[i][1] #naming dictonary that has the mapping, for example; 123 = topwear

print("Mapping image ID field in testing csv file to an image category") 
naming_dict_test = {}
fTe = open(".\\test.csv", "r")
fileContentsTest = fTe.read()
fileContentsTest = fileContentsTest.split('\n')
for i in range(len(fileContentsTest)-1):
  fileContentsTest[i] = fileContentsTest[i].split('\t')
  naming_dict_test[fileContentsTest[i][0]] = fileContentsTest[i][1] #naming dictonary that has the mapping, for example; 123 = topwear





print("Calculating how many article types there are in the training set")
typesTrain = naming_dict_train.values() 
types_set_train = set(typesTrain) #creates a set of range = total number of article types
counting_dict_train = {} #creates an qrray to keep track of how many times we have seen a particular article type, for all atricle types
for i in types_set_train:
  counting_dict_train[i] = 0 #intialize every article type to having 0 articles associated to it

print("Calculating how many article types there are in the testing set")
typesTest = naming_dict_test.values() 
types_set_test = set(typesTest) #creates a set of range = total number of article types
counting_dict_test = {} #creates an qrray to keep track of how many times we have seen a particular article type, for all atricle types
for i in types_set_test:
  counting_dict_test[i] = 0 #intialize every article type to having 0 articles associated to it





print("Mapping each training image to a product category")
for img in os.listdir(imgDIR): 
  imgName = img.split('.')[0] # converts name '0913209.jpg' --> '0913209'
  if (str(imgName) not in naming_dict_train): pass #avoids the case where we are analyzing a training image 
  else:
    label = naming_dict_train[str(imgName)] #maps the image to a product category, ie: label = naming_dict[123] = topwear
    counting_dict_train[label] += 1 #increments the number of times we've seen that kind of article
    path = os.path.join(imgDIR, img) #finds the exact path of that image
    #saves the name ARTICLETYPE-COUNT#
    saveName = ".\\labeled_train\\" + label + '-' + str(counting_dict_train[label]) + '.jpg'
    image_data = np.array(Image.open(path)) #load the data of that particular image into the model
    imageio.imwrite(saveName, image_data) #rewrites the saveName image as into our labeled image folder

print("Mapping each testing image to a product category")
for img in os.listdir(imgDIR): 
  imgName = img.split('.')[0] # converts name '0913209.jpg' --> '0913209'
  if (str(imgName) not in naming_dict_test): pass #avoids the case where we are analyzing a testing image 
  else:
    label = naming_dict_test[str(imgName)] #maps the image to a product category, ie: label = naming_dict[123] = topwear
    counting_dict_test[label] += 1 #increments the number of times we've seen that kind of article
    path = os.path.join(imgDIR, img) #finds the exact path of that image
    #saves the name ARTICLETYPE-COUNT#
    saveName = ".\\labeled_test\\" + label + '-' + str(counting_dict_test[label]) + '.jpg'
    image_data = np.array(Image.open(path)) #load the data of that particular image into the model
    imageio.imwrite(saveName, image_data) #rewrites the saveName image as into our labeled image folder





#Function to one hot encode all catoegory types
def label_img(name): 
  global array
  word_label = name.split('-')[0]
  if word_label == 'Topwear'    : array = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  if word_label == 'Bottomwear' : array = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  if word_label == 'Innerwear'  : array = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  if word_label == 'Bags'       : array = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  if word_label == 'Watches'    : array = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  if word_label == 'Jewellery'  : array = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  if word_label == 'Eyewear'    : array = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  if word_label == 'Wallets'    : array = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  if word_label == 'Shoes'      : array = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  if word_label == 'Sandal'     : array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  if word_label == 'Makeup'     : array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  if word_label == 'Fragrence'  : array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  elif word_label == 'Others'   : array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  
  return array






trainDIR = ".\\labeled_train\\"

print("Loading the training images")
def load_training_data():
  train_data = [] #create a training data array
  del train_data[:] #delete all elements
  for img in os.listdir(trainDIR):
    label = label_img(img) #return a one hot coded array for each image
    path = os.path.join(trainDIR, img) 
    img = Image.open(path)
    img = img.convert('L')
    img = img.resize((80,60), Image.ANTIALIAS)
    train_data.append([np.array(img), label]) #create a 2d array where element 0 is the image and element 1 is the np label array 

  shuffle(train_data)
  return train_data
  
train_data = load_training_data()

trainImages = []
del trainImages[:]
trainLabels = []
del trainLabels[:]

for entry in train_data:
  trainImages.append(entry[0])
  trainLabels.append(entry[1])

trainImages = np.array(trainImages)
trainImages = trainImages.reshape(len(trainImages), 80, 60, 1)
trainLabels = np.array(trainLabels)





testDIR = ".\\labeled_test\\"

print("Loading the testing images")
def load_testing_data():
  test_data = [] #create a training data array
  del test_data[:] #delete all elements
  for img in os.listdir(testDIR):
    label = label_img(img) #return a one hot coded array for each image
    path = os.path.join(testDIR, img) 
    img = Image.open(path)
    img = img.convert('L')
    img = img.resize((80,60), Image.ANTIALIAS)
    test_data.append([np.array(img), label]) #create a 2d array where element 0 is the image and element 1 is the np label array 

  shuffle(test_data)
  return test_data
  
test_data = load_testing_data()

testImages = []
del testImages[:]
testLabels = []
del testLabels[:]

for entry in test_data:
  testImages.append(entry[0])
  testLabels.append(entry[1])

testImages = np.array(testImages)
testImages = testImages.reshape(len(testImages), 80, 60, 1)
testLabels = np.array(testLabels)




print("Building the model")

model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape = (80, 60, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  #Dropout for regularization
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(13, activation='softmax')) #13 softmax for 13 classes

print("Compiling the model")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

print("Fitting the training set onto the model")
model.fit(trainImages, trainLabels, batch_size = 25, epochs=3 , verbose = 1)

print("Model Summary:")
model.summary()

"""
loss, acc = model.evaluate(trainImages, trainLabels, verbose = 0)
print("Accuracy on the training data is:")
print(acc * 100)
"""

""" 
loss, acc = model.evaluate(testImages, testLabels, verbose = 0)
print("Accuracy on the testing data is:")
print(acc * 100)
"""