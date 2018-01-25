########################################################################
##  text_modules.py
##  A collection of useful text modules to be used as plugins in larger code
##
##  by David Curry
##
#######################################################################

import nltk, re, pickle, random
from nltk.corpus import stopwords
from nltk import FreqDist
import sys
import numpy as np
import gensim.models.word2vec as w2v
from gensim.models.word2vec import Word2Vec
from itertools import islice
from keras.utils import np_utils
import keras
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier



def build_LSTM_model(x_train, n_vocab, HIDDEN_UNITS=10, HIDDEN_LAYERS=2, DROPOUT=0.1):
    model = Sequential()
    if HIDDEN_LAYERS == 1:
        model.add(LSTM(HIDDEN_UNITS, input_shape=(x_train.shape[1], n_vocab)))
    else:
        model.add(LSTM(HIDDEN_UNITS, input_shape=(x_train.shape[1], n_vocab), return_sequences=True))
    model.add(Dropout(DROPOUT))
    for layer in range(HIDDEN_LAYERS-1):
        model.add(LSTM(HIDDEN_UNITS))
        model.add(Dropout(DROPOUT))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model



def input_text(filename):
    '''Input : Text file path
       Output: text file'''

    print('\nLoading Text File:', filename)
    
    raw_text = open(filename).read()
    return raw_text


def clean_text(raw_text, keep_list, keep_upper = False):
    '''Input : Raw text file, keep uppercase bool, list of charachters to keep
       Output: Cleaned and processed text file'''

    print('\nCleaning Raw Text...')

    head = list(islice(raw_text, 100))
    print('\n Snippet of Raw Text:', head)

    cleaned_text = raw_text
    
    if not keep_upper:
        print('\nConverting all text to lowercase...')
        cleaned_text = raw_text.lower()
    
    cleaned_text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", cleaned_text)
    cleaned_text = re.sub(keep_list, "", cleaned_text)
    
    head = list(islice(cleaned_text, 100))
    print('\n Snippet of Cleaned Text:', head)
    
    return cleaned_text

def build_vocab(text):

    '''Input : Cleaned and charachter separated  text
       Output: unique charchter(vocab or features) dictionary and properties
       Also a mapping from unique charachter to its index.
       Returns: vocab, n_vocab, ix_to_char, char_to_ix
    '''
        
    vocab   = sorted(list(set(text)))
    n_vocab = int(len(vocab))
    
    print('\n Vocab List('+str(n_vocab)+' Unique Characters):\n', vocab)

    ix_to_vocab = {ix:char for ix, char in enumerate(vocab)}
    vocab_to_ix = {char:ix for ix, char in enumerate(vocab)}
    
    return vocab, n_vocab, ix_to_vocab, vocab_to_ix
    

def text_to_KerasRnn_input(text, vocab, n_vocab, seq_length, vocab_to_ix):

    '''Input : Cleaned and charachter separated text, vocab, num_vocab, sequence length, vocab dict
       Output: training arrays(x,y) for Keras RNN
       Returns: x_training_data, y_training_data
    '''
    
    print('\nConverting Text to Keras RNN Input Format with Sequence of', seq_length)
    print('Vocab length:', n_vocab)
    print("Total Characters: ", len(text))

    # cut the text in sequences of seq_length characters
    maxlen = seq_length
    step = 1
    sentences = []
    next_chars = []

    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    print('Number of Sequences:', len(sentences))

    print('Vectorization...')
    x = np.zeros((len(sentences), maxlen, n_vocab), dtype=np.bool)
    y = np.zeros((len(sentences), n_vocab), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, vocab_to_ix[char]] = 1
        y[i, vocab_to_ix[next_chars[i]]] = 1
    
    
    print('\nShape of input X Data:', x.shape[0], x.shape[1], x.shape[2])
    print('Shape of input Y Data:', y.shape[0], y.shape[1])
    
    return x, y


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, text, length, seq_length, n_vocab, ix_to_vocab, vocab_to_ix):

    ''' Generates new text from a learned Keras RNN
     Input : Trained Keras RNN model, how long we want the new gen text to be
     Output: New text of length
    '''

    print('\nGenerating New Text of Length', length,'\n')
    
    # pick a random seed
    start_index = random.randint(0, len(text) - seq_length - 1)

    # generate the seed text
    generated = ''
    sentence = text[start_index: start_index + seq_length]
    generated += sentence
    print('\n----- Generating with seed: "' + sentence + '"')

    print('----- End Seed -----')
    
    for i in range(length):
        x_pred = np.zeros((1, seq_length, n_vocab))
        for t, char in enumerate(sentence):
            x_pred[0, t, vocab_to_ix[char]] = 1.
            
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds)
        next_char = ix_to_vocab[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char


    print(generated)    
    
    print( "\nDone.")
    

def generate_text_diff_complexity(model_list, seed_text, length, seq_length, n_vocab, ix_to_vocab, vocab_to_ix):
    
    ''' Takes in a list of Keras models and returns a sequence of text in increasing complexity.
    INPUT: List of Keras models, seed text, and model parameters.  All models must be trained on same seq length and vocab
    OUTPUT: List of Generated Texts in increasing complexity 
    '''

    gen_text_list = []

    print('\nGenerating with seed: "' + sentence + '"')
        
    for model_file in model_list:
        
        print('\nGenerating New Text of Length', length,' for model', model_file)
        
        model = load_model(model_file)
        
        generated = ''
        sentence = seed_text
        generated += sentence
        
        for i in range(length):
            x_pred = np.zeros((1, seq_length, n_vocab))
            for t, char in enumerate(sentence):
                x_pred[0, t, vocab_to_ix[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds)
            next_char = ix_to_vocab[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
            
        #print(generated)
        
        gen_text_list.append([generated])

    print_text_diff_complexity(gen_text_list)
        
    return gen_text_list


def print_text_diff_complexity(text_list):

    ''' INPUT: list of gen text in order of increasing complexity
        Prints out each level of complexity in a clear way.
    '''

    for i,text in enumerate(text_list):

        print('\n\n==== Learning Stage #',i,'====')
        print(text[0])
        
    print('\nEvolution Finished...')

def sent_token(text):
    ''' Helper function for sentence_clean()'''
    text = nltk.sent_tokenize(text)
    return text

def sentence_clean(text):

    ''' Cleans sentences in a text and tokenizes words.
    Input: Text Body
    Output: List of tokenized sentences.
    '''

    new_text = []
    for sentence in text:
        sentence = sentence.lower()
        sentence = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", sentence)
        sentence = re.sub("[^a-z ]", "", sentence)
        sentence = nltk.word_tokenize(sentence)
        sentence = [word for word in sentence if len(word)>1] # exclude 1 letter words
        new_text.append(sentence)
    return new_text
  
def apply_all_sentence(text):
    ''' Helper function for sentence_clean()'''
    return sentence_clean(sent_token(text))








