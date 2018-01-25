########################################################################
##  keras_modules.py
##  A collection of useful modules to be used as plugins in larger code
##
##  by David Curry
##
#######################################################################

import keras
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

class TestCallback(keras.callbacks.Callback):

    '''
    Class for integration with Sci Kit and Keras Callabck class for epoch debugging.
    Input: (x_text, y_test)
    Output: Loss and Accuracy at each epoch 
    '''

    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def save_Model(my_model, path, title):

    ''' For saving Keras modules
        Input: Keras model, path for output,  title of output
        Returns: Keras model as h5 object
    '''
    
    json_model = my_model.model.to_json()
    open(path+'/'+title+'_architecture.json', 'w').write(json_model)
    my_model.model.save_weights(path+'/'+title+'_weights.h5', overwrite=True)
    return
    
def load_Model(path,title):

    ''' For loading Keras modules
        Input: path for load,  title to load
        Returns: Keras model 
    '''
    
    model = model_from_json(open(path+'/'+title+'_model_architecture.json').read())
    model.load_weights(path+'/'+title+'_model_weights.h5')
    return model






