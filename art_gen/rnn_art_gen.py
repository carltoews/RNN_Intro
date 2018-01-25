###############################
# My First RNN Text Generartion Script
# Uses Keras LSTM, modules in text_modules.py
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
from scipy import misc

# NN Modules
import keras
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier

# My modules
sys.path.insert(0,"/Users/HAL3000/Dropbox/coding/my_modules/")
import keras_modules as my_keras_modules
import w2v_modules as my_w2v_modules
import text_modules as my_text_modules
import image_modules as my_image_modules

# fix random seed for reproducibility
np.random.seed(7)


####################################################################
#  All tuneable parameters

# Run Optimizer
#optimize = True
optimize = False

# Run Image Generation
image_generate = True
#image_generate = False

# Training Image List
image_dir   = '/Users/HAL3000/Dropbox/coding/neural_nets/data/images/abs_exp/pollock/'
input_file = image_dir+'guggenheim_mural.jpg'

# Sequence length for RNN
SEQ_LENGTH = 100

# Generated Text Length
LENGTH = 200

### RNN Parameters ###
#retrain = False
retrain = True

HIDDEN_UNITS  = 10
HIDDEN_LAYERS = 2
DROPOUT       = 0.1
EPOCHS       = 2
BATCH_SIZE    = 128 

# RNN Hyperparameter Search Space
HIDDEN_UNITS_LIST  = [12, 64, 128, 256, 500, 700]
HIDDEN_LAYERS_LIST = [1,2,3,4,5]
DROPOUT_LIST       = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
EPOCHS_LIST        = (1, 5, 10, 20, 50, 100)
BATCH_SIZE_LIST    = (1, 10, 20, 64, 128)

grid_search_list = {
    'HIDDEN_UNITS':HIDDEN_UNITS_LIST,
    'HIDDEN_LAYERS':HIDDEN_LAYERS_LIST,
    'DROPOUT':DROPOUT_LIST,
    'epochs':EPOCHS_LIST,
    'batch_size':BATCH_SIZE_LIST}


# End tuneable parameters
#####################################################################


# Import the desired text(will split by charachter)
input_image = my_image_modules.import_image(input_file)

# Preprocess: lower case conversion, strip useless charachters, etc.
# cleaned_text = my_text_modules.clean_text(input_text, keep_list, keep_upper)

# # Define the unique charachter(feature or vocab) set.
# vocab, n_vocab, ix_to_vocab, vocab_to_ix = my_text_modules.build_vocab(cleaned_text)

# # Build the sequences that Keras RNN will train on.  Format for input is:
# # (number_of_sequences, length_of_sequence, number_of_features)
# x_train, y_train = my_text_modules.text_to_KerasRnn_input(cleaned_text, vocab, n_vocab, SEQ_LENGTH, vocab_to_ix)

# print('\nBuild model...')
# #model = my_text_modules.build_LSTM_model(x_train, HIDDEN_UNITS, HIDDEN_LAYERS, DROPOUT, n_vocab)

# def build_LSTM_model(DROPOUT=0.1, HIDDEN_LAYERS=1, HIDDEN_UNITS=10):
#     model = Sequential()
#     if HIDDEN_LAYERS == 1:
#         model.add(LSTM(HIDDEN_UNITS, input_shape=(x_train.shape[1], n_vocab)))
#     else:
#         model.add(LSTM(HIDDEN_UNITS, input_shape=(x_train.shape[1], n_vocab), return_sequences=True))
#     model.add(Dropout(DROPOUT))
#     for layer in range(HIDDEN_LAYERS-1):
#         model.add(LSTM(HIDDEN_UNITS))
#         model.add(Dropout(DROPOUT))
#     model.add(Dense(n_vocab, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam')
#     model.summary()
#     return model

# # define the checkpoint and callbacks
# mydir = datetime.datetime.now().strftime('%m-%d_%H-%M')
# os.makedirs("logs/"+mydir+"/")
# filepath = "logs/"+mydir+"/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# tb_callback = keras.callbacks.TensorBoard(log_dir='./logs/fake_news/{}'.format(time()),
#                                           histogram_freq=0, write_graph=True,
#                                           write_images=False)
# callbacks_list = [checkpoint, tb_callback]


# # Optimise the RNN Hyperparameters
# if optimize:
#     # create sci kit model from Keras model
#     #my_model = KerasClassifier(build_fn = my_text_modules.build_LSTM_model(x_train, HIDDEN_UNITS, HIDDEN_LAYERS, DROPOUT, n_vocab),
#     #                           epochs=3, batch_size=10)
    
#     my_model = KerasClassifier(build_fn = build_LSTM_model)
    
#     print('\n====== Optimization with SK Grid Search ========')
#     print(grid_search_list)
    
#     grid = GridSearchCV(estimator = my_model, param_grid = grid_search_list, n_jobs=-1, cv=3)
#     grid_result = grid.fit(x_train, y_train)

#     # summarize results
#     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#     for params, mean_score, scores in grid_result.grid_scores_:
#         print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
    

# # Fit the RNN
# if not os.path.exists('weights/rnn_alice.h5') or retrain:
#     print('\n ==== Training Keras NN ====')
#     print('Epochs:', EPOCHS, '\nbatch size:', BATCH_SIZE)
#     my_model = KerasClassifier(build_fn = build_LSTM_model)
#     my_model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list)
#     my_model.save('weights/rnn_alice.h5')
# else:
#     print('\nImporting Keras NN Model...')
#     my_model = load_model('logs/weights-improvement-02-1.8823.hdf5')


# # Generate some new text(trained RNN model, desried new text length, vocab size, two vocab dicts)

# if text_generate:
#     my_text_modules.generate_text(my_model, cleaned_text, LENGTH, SEQ_LENGTH,
#                                   n_vocab, ix_to_vocab, vocab_to_ix)



    
      
