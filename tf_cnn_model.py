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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

TRAIN_DIR = '../data/train/'
TEST_DIR = '../data/test/'
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
N_CLASSES = len(FISH_CLASSES)
ROWS = 90
COLS = 160
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


#--------- TF CNN functions
def conv2D(x, W, b, strides=1):
	"""Apply 2D conv including ReLU"""
	conv = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	conv = tf.nn.bias_add(conv, b)
	return conv

def max_pool_2D(x, k=2):
	"""Apply 2D max pooling"""
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, 2, 2, 1], padding='SAME')


#------- Building TF graph
def tf_inputs():
	x = tf.placeholder("float", [None, ROWS, COLS, CHANNELS])
	y = tf.placeholder("float", [None, N_CLASSES])

	# Weights and biases - initialize with normal random numbers
	weights = {
		'conv1': tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.1)),
		'conv2': tf.Variable(tf.truncated_normal([5, 5, 32, 32], stddev=0.1)),
		'conv3': tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1)),
		'conv4': tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1)),
		'conv5': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1)),
		'conv6': tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1)),
		'conv7': tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1)),
		'conv8': tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1)),
		'fc1': tf.Variable(tf.truncated_normal([6*10*256, 256], stddev=0.1)),
		'fc2': tf.Variable(tf.truncated_normal([256, 64], stddev=0.1)),
		'output': tf.Variable(tf.truncated_normal([64, N_CLASSES], stddev=0.1))
	}
	biases = {
		'conv1': tf.Variable(tf.constant(0.1, shape=[32])),
		'conv2': tf.Variable(tf.constant(0.1, shape=[32])),
		'conv3': tf.Variable(tf.constant(0.1, shape=[64])),
		'conv4': tf.Variable(tf.constant(0.1, shape=[64])),
		'conv5': tf.Variable(tf.constant(0.1, shape=[128])),
		'conv6': tf.Variable(tf.constant(0.1, shape=[128])),
		'conv7': tf.Variable(tf.constant(0.1, shape=[256])),
		'conv8': tf.Variable(tf.constant(0.1, shape=[256])),
		'fc1': tf.Variable(tf.constant(0.1, shape=[256])),
		'fc2': tf.Variable(tf.constant(0.1, shape=[64])),
		'output': tf.Variable(tf.constant(0.1, shape=[N_CLASSES]))
	}

	return(x, y, weights, biases)

def tf_graph(x, y, weights, biases, learning_rate=0.1):
	# Convolutional and max pooling layers
	conv1 = conv2D(x, weights['conv1'], biases['conv1'])
	conv2 = conv2D(conv1, weights['conv2'], biases['conv2'])
	pool1 = max_pool_2D(conv2)
	conv3 = conv2D(pool1, weights['conv3'], biases['conv3'])
	conv4 = conv2D(conv3, weights['conv4'], biases['conv4'])
	pool2 = max_pool_2D(conv3)
	conv5 = conv2D(pool2, weights['conv5'], biases['conv5'])
	conv6 = conv2D(conv5, weights['conv6'], biases['conv6'])
	pool3 = max_pool_2D(conv5)
	conv7 = conv2D(pool3, weights['conv7'], biases['conv7'])
	conv8 = conv2D(conv7, weights['conv8'], biases['conv8'])
	pool4 = max_pool_2D(conv8)

	# Fully connected layers
	pool4_flat = tf.reshape(pool4, [-1, 6*10*256])
	fc1 = tf.nn.relu(tf.add(tf.matmul(pool4_flat, weights['fc1']), biases['fc1']))
	fc1_drop = tf.nn.dropout(fc1, keep_prob=0.5)
	fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, weights['fc2']), biases['fc2']))
	fc2_drop = tf.nn.dropout(fc2, keep_prob=0.5)

	# Output layer 
	pred = tf.add(tf.matmul(fc2_drop, weights['output']), biases['output'])
	y_pred = tf.nn.softmax(pred)

	# Evaluation
	cost = tf.nn.softmax_cross_entropy_with_logits(pred, y)
	opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	pred_correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(pred_correct, "float"))

	return(y_pred, cost, opt, pred_correct, accuracy)



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

	# Load data into numpy arrays and dump to pkl file
	# for i, im in enumerate(files): 
	#     X_all[i] = read_image(TRAIN_DIR+im)
	#     if i%1000 == 0: print('Processed {} of {}'.format(i, len(files)))

	# with open('./data/X_train_data.pkl', 'wb') as f:
	# 	pickle.dump(X_all, f, protocol=pickle.HIGHEST_PROTOCOL)

	print("Loading training data from pickle file...")
	with open('../data/X_train_data.pkl', 'rb') as f:
		X_all = pickle.load(f)

	# One-hot encode the labels (see labels_dict for mapping)
	idx = 0
	labels_dict = {}
	for fish in FISH_CLASSES:
		labels_dict[fish] = idx
		y_all[y_all==fish] = idx
		idx += 1

	y_all = y_all.reshape([-1, 1]).astype(np.int)
	lb = LabelBinarizer()
	y_all = lb.fit_transform(y_all)

	# Split into train and validation sets
	X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all)

	# Load test data and dump to pkl file
	test_files = [im for im in os.listdir(TEST_DIR)]
	# test = np.ndarray((len(test_files), ROWS, COLS, CHANNELS), dtype=np.uint8)
	# for i, im in enumerate(test_files):
	# 	test[i] = read_image(TEST_DIR+im)

	# with open('./data/test_data.pkl', 'wb') as f:
	# 	pickle.dump(test, f, protocol=pickle.HIGHEST_PROTOCOL)

	print("Loading test data from pickle file...")
	with open('../data/test_data.pkl', 'rb') as f:
		test = pickle.load(f)


	#---------------------- Part 2: Tensorflow CNN model
	learning_rate = 0.01
	batch_size = 100
	n_inputs = X_train.shape[0]
	n_batches = int(n_inputs / batch_size)
	n_val_batches = int(X_val.shape[0] / batch_size)
	n_test_batches = int(test.shape[0] / batch_size)

	# Build TF graph
	x, y, weights, biases = tf_inputs()
	y_pred, cost, opt, pred_correct, accuracy = tf_graph(x, y, weights, biases, learning_rate)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	# Run model
	# Try this to limit CPU usage - http://stackoverflow.com/questions/38615121/limit-tensorflow-cpu-and-memory-usage
	with tf.Session() as sess:
		sess.run(init)

		t0 = time.time()

		print("#-------- Beginning training....")
		for i in range(n_batches):
			sess.run([opt, cost], feed_dict={x: X_train[i:i+batch_size, :, :, :], y: y_train[i:i+batch_size, :]})
			print("Trained for batch %d of %d, time elapsed %f seconds" % (i+1, n_batches, time.time()-t0))

			# # Take break to avoid CPU crash (need to limit threads TF can use instead)
			# if (i != 0) & (i % 10 == 0):
			# 	print("Sleeping for 60s...")
			# 	time.sleep(60)

		save_path = saver.save(sess, "./CNN_model")

		print("#-------- Testing on validation set... ")
		val_cost = 0
		for i in range(n_val_batches):
			val_cost += sess.run(cost, feed_dict={x: X_val[i:i+batch_size, :, :, :], y: y_val[i:i+batch_size, :]})
			print("Tested for batch %d of %d" % (i+1, n_val_batches))
		val_cost = np.sum(val_cost, dtype=np.float32) / X_val.shape[0]
		print("Validation set cost = %f" % val_cost)


		# Predict on test set
		print("#-------- Making predictions on test set...")
		test_preds = []
		for i in range(n_test_batches):
			test_preds.append(sess.run(y_pred, feed_dict={x: test[i:i+batch_size, :, :, :]}))
			print("Tested for batch %d of %d" % (i+1, n_test_batches))

		sess.close()

		#-------------------- Part 3: Test submission
		test_preds = np.concatenate(test_preds)
		submission = pd.DataFrame(test_preds, columns=FISH_CLASSES)
		submission.insert(0, 'image', test_files)
		print(submission.head())
		submission.to_csv('./submission.csv', index=False)


if __name__ == "__main__":
	random.seed(0)
	main()
