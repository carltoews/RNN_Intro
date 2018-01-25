########################################################################
##  image_modules.py
##  A collection of useful modules to be used as plugins in larger code
##
##  by David Curry
##
#######################################################################

# import dependencies
import keras
import csv as csv
import numpy as np
import pandas as pd
import pylab as py
import re, pickle, sys, os
from time import time
import matplotlib.pyplot as plt
from scipy import misc

def import_image(file, verbose=True):
    '''Imports image using scipy
       Returns: Numpy array of image'''

    image = misc.imread(file)
    
    if verbose:
        print('\nImported Image', file)
        print('Image Shape, Type:',image.shape, image.dtype)
        #print(image)
        # Note:  Numpy shape for color arrays will ahve 3rd dimension for color layers(red, green, blue).
        #plt.imshow(image)
        #plt.show()
    return image


