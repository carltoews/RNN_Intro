######################
# miscellaneous methods
#
# by David Curry
#####################

import numpy as np
import matplotlib.pyplot as plt

def pie_chart(frac, labels):
    ''' Takes in:
            num_cat: # of colors in the pier chart
            frac: % or each category
    '''

    explode = (0, 0.05, 0, 0)
    
    plt.pie(fracs, labels=labels, autopct='%1.1f%%')

    plt.savefig('pie_dog.png')
    
labels = 'No Dog', 'Dog in Pic'
fracs = [7, 93]    

pie_chart(fracs, labels)
