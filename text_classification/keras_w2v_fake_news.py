###############################
# Using Keras and word2vec for feature reduction.
# Binary Text Classification
# By David Curry
# 11/2017
###############################


# import dependencies
import csv as csv
import numpy as np
import pandas as pd
import pylab as py
import re, pickle, sys, os, multiprocessing
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cross_validation import KFold
import sklearn.metrics as skmetrics
import gensim.models.word2vec as w2v
from gensim.models.word2vec import Word2Vec
import pickle

# NN Modules
import keras
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# My modules
sys.path.insert(0,"/Users/HAL3000/Dropbox/coding/my_modules/")
import keras_modules as my_keras_modules
import w2v_modules as my_w2v_modules

# fix random seed for reproducibility
np.random.seed(123)

###################################

retrain = True
#retrain = False
 
### Word2Vec Parameters ###
num_features = 300
min_word_count = 10
num_workers = multiprocessing.cpu_count()
context_size = 7
downsampling = 1e-3

w2v_parameters = {'num_features':num_features, 'min_word_count':min_word_count,
                  'num_workers':num_workers, 'context_size':context_size, 'downsampling':downsampling}

### Keras NN Parameters ###
num_epochs  = 2
num_batches = 10 
num_layers  = 1
num_hidden_units = 8

####################################

# Load the fake and real news
fake_data = pd.read_csv('../fakeNews_classifier/data/fake.csv')
fake_data = fake_data[fake_data.language == 'english']
fake_data.dropna(axis=0, inplace=True, subset=['text'])
fake_data.reset_index(drop=True,inplace=True)
fake_data.describe()

# Now the Real Data
real_data = pd.read_csv('../fakeNews_classifier/data/real_news.csv')
real_data = real_data[fake_data.language == 'english']
real_data.dropna(axis=0, inplace=True, subset=['text'])
real_data.reset_index(drop=True,inplace=True)

# Add category names(Fake, Real) to their respective dataset
fake_data['class'] = 0
real_data['class'] = 1

fake_and_real_data = pd.concat([fake_data, real_data]).sample(frac=1.0)

# Train w2v
if not os.path.exists('weights/w2v_fake_dict.p'):

    #my_w2v = my_w2v_modules.create_w2v_cv(train_text, w2v_parameters, 'fake')

    my_w2v = my_w2v_modules.create_w2v_nltk_sentence(fake_and_real_data, w2v_parameters, 'fake')
    
else:    
    print('\nLoading Word2Vec Results...')
    #all_word_vectors_matrix = pickle.load(open('weights/w2v_fake_matrix.p', 'rb'))
    #vocab = pickle.load(open('weights/w2v_fake_vocab.p', 'rb'))
    #vocab_index = pickle.load(open('weights/w2v_fake_index.p', 'rb'))
    my_w2v = pickle.load(open('weights/w2v_fake_dict.p', 'rb'))
    
 
# Output shape
print('\nW2V Weight Dict Shape:', len(my_w2v), len(my_w2v[0]))

# Create the new features by averaging over w2v features
w2v_mean = my_w2v_modules.MeanEmbeddingVectorizer(my_w2v)
w2v_mean_features = w2v_mean.transform(train_text)

print('\nNew w2v Features Dimensions:',  w2v_mean_features.shape)

# Now we need to make sure this is in the right form for our Keras NN and inti the NN.
# Define train/test features and class labels
# Convert scipy sparse matrix to numpy

x_train, x_test, y_train, y_test = train_test_split(w2v_mean_features, train_y, test_size=0.33, random_state=42)

num_features = w2v_mean_features.shape[1]

# Create the Keras model
def create_model():
    model = Sequential()
    model.add(Dense(num_hidden_units, input_dim=num_features, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('\n==== Keras Architecture ====', model.summary())
    return model

# For visualizing and tracking performance during runtime
# Use : tensorboard --logdir=logs/
tb_callback = keras.callbacks.TensorBoard(log_dir='./logs/fake_news/{}'.format(time()),
                                          histogram_freq=0, write_graph=True,
                                          write_images=False)

# create sci kit model from Keras model
my_model = KerasClassifier(build_fn = create_model,
                           epochs=3, batch_size=10,
                           callbacks=[tb_callback])


if not os.path.exists('weights/1_layer_'+str(num_epochs)+'_epochs_weights.h5') or retrain:
    print('\nTraining Keras NN...')
    #os.remove('weights/10_layer_model.h5')
    my_model.fit(x_train, y_train)
    keras_modules.save_Model(my_model, 'weights', '1_layer_'+str(num_epochs)+'_epochs')
    #my_model.model.save('weights/10_layer_model.h5')
else:
    print('\nImporting Keras NN Model...')
    #my_model = keras.models.load_model('weights/10_layer_model.h5')
    my_model = keras_modules.load_Model('weights', '1_layer_'+str(num_epochs)+'_epochs') # my own function


print('\nEvaluating Keras NN Model...')

predictions = np.round(my_model.predict(x_test))

target_names = ['class 0: Fake', 'class 1: Real']

print(skmetrics.classification_report(y_test, predictions, target_names=target_names))
print (skmetrics.confusion_matrix(y_test, predictions))









