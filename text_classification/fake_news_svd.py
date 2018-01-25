###############################
# Using SVD on fake news text for Keras NN.
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
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cross_validation import KFold
import sklearn.metrics as skmetrics
from sklearn.decomposition import TruncatedSVD

# NN Modules
import keras
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# My modules
sys.path.insert(0,"/Users/HAL3000/Dropbox/coding/my_modules/")
import keras_modules as my_keras_modules

# fix random seed for reproducibility
np.random.seed(7)

### Keras NN Parameters
retrain = True
#retrain = False

num_epochs  = 30
num_batches = 1 
num_layers  = 1
num_hidden_units = 256
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

if not os.path.exists('cv_embeddings.p'):
    count_vect = CountVectorizer(stop_words="english")
    print('\nCreating Word Embeddings...')
    realfake_matrix_CV = count_vect.fit_transform(fake_and_real_data['text'].values.astype('U'))
    pickle.dump(realfake_matrix_CV, open('cv_embeddings.p', 'wb')) 
else:
    print('\nImporting Word Embeddings...')
    realfake_matrix_CV = pickle.load(open('cv_embeddings.p', 'rb'))

print('\nNews dataset dimensions BEFORE SVD:', realfake_matrix_CV.shape)

# Perform the SVD
my_svd = TruncatedSVD(n_components = 20)
realfake_matrix_SVD = my_svd.fit_transform(realfake_matrix_CV)

print('\nNews dataset dimensions AFTER SVD:', realfake_matrix_SVD.shape)

print('\nVariance in SVD components:', my_svd.explained_variance_ratio_)
    
# Define train/test features and class labels
# Convert scipy sparse matrix to numpy
train_text = realfake_matrix_SVD
train_y    = fake_and_real_data['class']

x_train, x_test, y_train, y_test = train_test_split(train_text, train_y, test_size=0.33, random_state=42)

print('\nNews dataset dimensions:', realfake_matrix_SVD.shape)

num_features = realfake_matrix_SVD.shape[1]

# Create the Keras model
def create_model():
    model = Sequential()
    model.add(Dense(num_hidden_units, input_dim=num_features, activation='relu'))
    model.add(Dense(num_hidden_units, activation='relu'))
    model.add(Dense(num_hidden_units, activation='relu'))
    #model.add(Dense(num_hidden_units, activation='relu'))
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
                           epochs=num_epochs,
                           batch_size=10,
                           #callbacks=[my_keras_modules.TestCallback((x_test, y_test))])
                           callbacks=[tb_callback])



if not os.path.exists('weights/1_layer_'+str(num_epochs)+'_epochs_weights.h5') or retrain:

    print('\nTraining Keras NN...')

    my_model.fit(x_train, y_train, validation_data=(x_test, y_test))
        
    my_keras_modules.save_Model(my_model, 'weights', '1_layer_'+str(num_epochs)+'_epochs')

else:
    print('\nImporting Keras NN Model...')
    #my_model = keras.models.load_model('weights/10_layer_model.h5')
    my_model = my_keras_modules.load_Model('weights', '1_layer_'+str(num_epochs)+'_epochs') # my own function

    
print('\nEvaluating Keras NN Model...')    

predictions = np.round(my_model.predict(x_test))

target_names = ['class 0: Fake', 'class 1: Real']

print(skmetrics.classification_report(y_test, predictions, target_names=target_names))
print (skmetrics.confusion_matrix(y_test, predictions))

