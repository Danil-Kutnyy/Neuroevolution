
#Compiler
import time
from random import randint
import random
import numpy as np
import re
import math
import networkx as nx
import json
import multiprocessing
from GenPile_2 import NN
import tensorflow as tf
from tensorflow import keras
from tensorflow_model import nn_to_tensorflow
import os

neural_netwrok_genome = "ACTG" #genetic code here

dir_path = os.path.dirname(os.path.realpath(__file__))
with open('{url}/boost_perfomnc_gen/default_gen_284.json'.format(url=dir_path), 'r') as file:
	list_i = json.load(file)
	neural_netwrok_genome = list_i[0][3] # list_i is a 2-d tensor. 
	#First dimension - population of inividual. If there was 10 NNs in a population, dimesnion = 10
	#Second dimension - have such form: [«total number of cells in nn», «number of divergent cell names», «number of layer connections», «genome»],
	#so if you want to get genome from a json file, use list_i["NNs index number"][3]




neural_network_object = NN(neural_netwrok_genome, max_time=12)
neural_network_object.develop()

nn_to_tensorflow(neural_network_object.nn, shape=[[28,28]])

from model_tmp import model
tf.keras.utils.plot_model(model, "test_model.png",show_shapes=True)
if neural_network_object.learning_rate == []:
	neural_network_object.learning_rate = [0.01]
optimizer = tf.keras.optimizers.Adam(learning_rate=neural_network_object.learning_rate[0])
model.compile(optimizer='adam',
			              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
			              metrics=['accuracy'])
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
model.summary()
try:
	tf.config.run_functions_eagerly(False)
	model.fit(train_images, train_labels, epochs=1, batch_size=32, verbose=1)
except ValueError:
	tf.config.run_functions_eagerly(True)
	model.fit(train_images, train_labels, epochs=1, batch_size=32, verbose=1)
