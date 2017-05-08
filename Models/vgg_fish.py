import os
import sys
import time
import cv2
import pdb
import random
import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import log_loss

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')

TRAIN_DIR = '../data/train/'
TEST_DIR = '../data/test/'
MODEL_DIR = '../trained_models/'
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
N_CLASSES = len(FISH_CLASSES)
ROWS = 224
COLS = 224
CHANNELS = 3

#---------- Functions for reading in images
def get_images(fish):
	"""Load files from train folder"""
	fish_dir = TRAIN_DIR + '{}'.format(fish)
	images = [fish+'/'+im for im in os.listdir(fish_dir)]
	return images

def read_image(src):
	"""Read and resize individual images"""
	im = cv2.imread(src, cv2.IMREAD_COLOR)
	im = cv2.resize(im, (COLS, ROWS), interpolation=cv2.INTER_CUBIC)  # resizing images
	return im

#---------- Helper functions
def center_normalize(x):
	return (x - K.mean(x)) / K.std(x)

def pop(model):
    '''Removes a layer instance on top of the layer stack.'''
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    else:
        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-1].outbound_nodes = []
            model.outputs = [model.layers[-1].output]
        model.built = False

    return model


def VGG_16(weights_path=None):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3, 224, 224)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='softmax'))

	if weights_path:
		model.load_weights(weights_path)

	return model

def transfer_model(VGG_model):
	#Remove the last two layers to get the 4096D activations - feature representation
	VGG_model = pop(VGG_model)
	VGG_model = pop(VGG_model)

	for l in VGG_model.layers:
		l.trainable = False

	model = Sequential()
	model.add(VGG_model)
	model.add(Dropout(0.5))
	model.add(Dense(N_CLASSES, activation='softmax'))

	return model



def main():

	#------------------ Part 1: Loading data and splitting into train and test
	files = []
	y_all = []

	# Get list of files and count per fish type
	for fish in FISH_CLASSES:
		fish_files = get_images(fish)
		files.extend(fish_files)

		y_fish = np.tile(fish, len(fish_files))
		y_all.extend(y_fish)
		print("{0} photos of {1}".format(len(fish_files), fish))

	y_all = np.array(y_all)
	X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)


	#----------- Test image for VGG net
	test_image = X_all[0, :, :, :].transpose((2, 0, 1)).reshape(1, CHANNELS, ROWS, COLS)

	model = VGG_16(MODEL_DIR + 'vgg16_weights.h5')
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy')
	out = model.predict(test_image)
	print np.argmax(out)


	# print("Loading training data from pickle file...")
	# with open('./data/X_train_data.pkl', 'rb') as f:
	# 	X_all = pickle.load(f)

	# # One-hot encode the labels (see labels_dict for mapping)
	# idx = 0
	# labels_dict = {}
	# for fish in FISH_CLASSES:
	# 	labels_dict[fish] = idx
	# 	y_all[y_all==fish] = idx
	# 	idx += 1

	# y_all = y_all.reshape([-1, 1]).astype(np.int)
	# lb = LabelBinarizer()
	# y_all = lb.fit_transform(y_all)

	# # Split into train and validation sets
	# X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2)

	# # Load test data
	# test_files = [im for im in os.listdir(TEST_DIR)]
	# # test = np.ndarray((len(test_files), ROWS, COLS, CHANNELS), dtype=np.uint8)
	# # for i, im in enumerate(test_files):
	# # 	test[i] = read_image(TEST_DIR+im)

	# # with open('./data/test_data.pkl', 'wb') as f:
	# # 	pickle.dump(test, f, protocol=pickle.HIGHEST_PROTOCOL)

	# print("Loading test data from pickle file...")
	# with open('./data/test_data.pkl', 'rb') as f:
	# 	test = pickle.load(f)



	# #--------------------- Part 2: CNN model with Keras
	# model = keras_cnn()
	# early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')        
	# model.fit(X_train, y_train, batch_size=64, nb_epoch=1, validation_split=0.2, verbose=1, 
	# 	shuffle=True, callbacks=[early_stopping])




if __name__ == "__main__":
	main()



