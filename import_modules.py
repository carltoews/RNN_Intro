##########################################
#  import_modules.py
#
#  Useful modules that are frequently used
#
# By David Curry,  circa 2017
#
##########################################

import csv as csv
import numpy as np
import pandas as pd
import pylab as py
import sys
import matplotlib.pyplot as plt
 

def import_modules():
    '''
    defines modules to be improrted
    '''

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
    


    
    
