###############################
# Intro Keras NN Exercices
#
# By David Curry
# 11/2017
###############################


# import dependencies
import csv as csv
import numpy as np
import pandas as pd
import pylab as py
import re, pickle, sys, os
import nltk
from pprint import *
from sklearn.manifold import TSNE
import multiprocessing
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
import sklearn.metrics as skmetrics

# NN Modules
import keras
from keras.models import Sequential
from keras.layers import Dense

# fix random seed for reproducibility
np.random.seed(7)

# In this tutorial, we are going to use the Pima Indians onset of diabetes dataset. This is a standard machine learning dataset from the UCI Machine Learning repository. It describes patient medical record data for Pima Indians and whether they had an onset of diabetes within five years.

# load pima indians dataset
dataset = np.loadtxt("data/pima-indians-diabetes.data.csv", delimiter=",")

# split into input (X) and output (Y) variables.  This dataset has 8 features
x = dataset[:,0:8]
y = dataset[:,8]

print('\nInput Dataset dimensions:', x.shape)

# For Keras NNs we need rows = examples and columns = features
num_features = x.shape[1]
num_examples = x.shape[0]

# create Keras NN model
model = Sequential()

# 1st hidder layer: # of nuerons, # of input features, neuron activation function
model.add(Dense(8, input_dim=num_features, activation='relu'))
#model.add(Dense(8, activation='relu'))
# Output layer: 1 nueron with sigmoid for binary classification
model.add(Dense(1, activation='sigmoid'))

print('\n ==== Keras NN Architecture ====')
model.summary()

# Compile model:   Loss function,  SGD algorith, output metrics for binary classification
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# For visualizing and tracking performance during runtime
tb_callback = keras.callbacks.TensorBoard(log_dir='./Graph/model_1/',
                                          histogram_freq=0, write_graph=True,
                                          write_images=False)

# Fit the model
model.fit(x, y, epochs=150, batch_size=10,
          callbacks=[tb_callback])

# evaluate the model on training set just for dummy check
scores = model.evaluate(x, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


